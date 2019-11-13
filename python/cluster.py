import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import math
import time
import sys
import os
import random

from data_api import load_corpus


np.set_printoptions(threshold=np.inf)


def get_layer_output(layers, layer_name, img):
    # format for tf
    curr = np.array([img, ])
    # execute the layers
    for layer in layers:
        # run layer
        curr = layer(curr)

        if layer.name == layer_name:
            return curr

    return None


def load_image(file_name):
    img = np.float32(cv2.imread(file_name))
    # Remove two channels
    img = np.delete(img, [1, 2], axis=2)
    img = img / 255

    return img.squeeze()


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def save_imgs(imgs, name, ids=None):
    if not plt.get_fignums():
        plt.figure()

    row_size = 10.0
    # if len(imgs) <= 20:
    #     row_size = 4.0
    for i, img in enumerate(imgs):
        plt.subplot(math.ceil(len(imgs) / row_size), int(row_size), i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        if ids is not None:
            plt.ylabel(str(ids[i]))
        plt.axis('off')

    plt.savefig(name + '.png')
    plt.clf()
    # plt.show()


def save_img(img, name):
    if not plt.get_fignums():
        plt.figure()
    plt.imshow(img.squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(name + '.png')
    plt.clf()


def save_bar(a, name):
    outlier_idx = get_outliers(a)
    a_outliers = []
    for i, item in enumerate(a):
        if i in outlier_idx:
            a_outliers.append(item)
        else:
            a_outliers.append(0)

    # thresh_idx = get_threshold(a, 0.05)
    # print('thresh_idx', thresh_idx)
    # a_thresh = []
    # for i, item in enumerate(a):
    #     if i in thresh_idx:
    #         a_thresh.append(item / 2)
    #     else:
    #         a_thresh.append(0)

    if not plt.get_fignums():
        plt.figure()
    plt.bar(np.arange(len(a)), a)
    plt.bar(np.arange(len(a_outliers)), a_outliers)
    # plt.bar(np.arange(len(a_thresh)), a_thresh)
    plt.savefig(name + '.png')
    plt.clf()


def format(img):
    # Make 225, 225 shape
    img = cv2.resize(img, (225, 225))
    # invert black and white
    img = 1 - img
    img = np.asarray(img).astype(np.float32).reshape((225, 225, 1))
    return img


def get_outliers(a):
    a_sorted = sorted(a)
    # only look at top since sparse
    q1, q3 = np.percentile(a_sorted, [64, 88])
    iqr = q3 - q1
    # lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_idx = []
    for i, item in enumerate(a):
        if item > upper_bound:
            outlier_idx.append(i)

    return outlier_idx


def get_threshold(a, thresh):
    outlier_idx = []
    for i, item in enumerate(a):
        if item > thresh:
            outlier_idx.append(i)

    return outlier_idx


def make_avg_img(imgs):
    # avg_cluster_img = np.zeros(imgs[0].shape)
    # for i, img in enumerate(imgs):
    #     # sum images with more weight on higher matches
    #     avg_cluster_img += img * (len(imgs) - i)

    avg_cluster_img = imgs[0].copy().squeeze()
    for i, img in enumerate(imgs[1:]):
        avg_cluster_img_8U = np.uint8(avg_cluster_img * 255).squeeze()
        img_8U = np.uint8(img * 255).squeeze()
        # try to align img with avg
        img_aligned = img_8U
        # try:
        #     img_aligned = alignImages(avg_cluster_img_8U, img_8U)
        # except cv2.error as e:
        #     # cannot align
        #     pass

        # take average of current and new image (weighted so that each image is averaged equally)
        avg_cluster_img += img_aligned * (1.0 / (i + 1)) / 255
        cv2.normalize(avg_cluster_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # plt.figure()
        # plt.imshow(avg_cluster_img.squeeze(), cmap='gray')
        # plt.axis('off')
        # plt.title('Updated')
        # plt.show()

    return avg_cluster_img


def alignImages(im1_gray, im2_gray):
    '''
    Modified From: https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    '''
    # Find size of image1
    sz = im1_gray.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 1)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2_gray, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned


def save_results(imgs_dict, acts_dict, centers, path):
    num_matches = 100
    # save images for visualizing clusters
    print('Save top matches and bar charts for each cluster')
    avg_cluster_imgs = []
    stats_file = open(path + 'stats.txt', 'w')
    total_imgs = 0
    for i, center in enumerate(centers):
        print('Saving results for cluster ' + str(i))
        matches_acts = np.array(acts_dict[i])
        matches_img = np.array(imgs_dict[i])
        target = center

        # get the error for all the acts matching this cluster and sort
        error = np.sum((matches_acts - target) ** 2, 1)
        sort_idx = np.argsort(error)

        # print the image part matches
        top_matches = []
        top_matches_ids = []
        num = min(num_matches, len(matches_img))
        for idx in range(num):
            match_index = sort_idx[idx]
            top_match = matches_img[match_index]
            top_matches.append(top_match)
            top_matches_ids.append(idx)

        # make average image of top matches
        avg_cluster_img = make_avg_img(top_matches)
        avg_cluster_imgs.append(avg_cluster_img)

        save_imgs(top_matches, path + 'top_matches_' + ('0' if i < 10 else '') + ('00' if i < 100 else '') + str(i), top_matches_ids)

        # get min distance between centers
        center_dists = []
        for j, center_ in enumerate(centers):
            dist = length(center - center_)
            if i is not j:
                center_dists.append(dist)
        sort_idx = np.argsort(center_dists)
        closest_center_dists = []
        for idx in sort_idx[:3]:
            closest_center_dists.append((idx, round(center_dists[idx], 5)))

        # get avg distance to points for each center
        avg_dist_to_center = np.mean(error)

        # save stats to text file
        stats_file.write('Center ' + str(i) + ': closest_centers(' + str(closest_center_dists) + '), num_assignments(' + str(len(acts_dict[i])) + '), avg_dist_to_center(' + str(round(avg_dist_to_center, 5)) + ')\n')

        # save bar charts for acts
        file_name = ('0' if i < 10 else '') + ('00' if i < 100 else '') + str(i)
        save_bar(center, path + 'charts_' + file_name)
        total_imgs += len(acts_dict[i])

    stats_file.write('\nTotal: ' + str(total_imgs) + '\n')
    stats_file.close()
    save_imgs(avg_cluster_imgs, path + 'top_matches_avg')
    print('Finished saving')


def find_k(imgs, acts, k_range, path):
    # run kmeans at different k's
    print('Start K-means...')
    start, stop = k_range
    avg_dist_by_k = []
    cluster_stats_by_k = []
    t0 = time.time()
    for k in range(start, stop):
        print('Trying K-means for k=' + str(k))
        km = KMeans(n_clusters=k, init='k-means++')
        num = min(500 + (k * 10), len(acts))
        assignments = km.fit_predict(acts[0:])
        centers = km.cluster_centers_
        inertia = km.inertia_

        avg_dist_by_k.append(inertia)
        t1 = time.time()
        print('Finished in ' + str(round(t1 - t0, 1)) + 's')
        t0 = t1

    # make the directory to store these files
    try:
        os.mkdir(path)
    except OSError:
        print('Creation of the directory %s failed' % path)
    else:
        print('Successfully created the directory %s ' % path)

    if not plt.get_fignums():
        plt.figure()
    plt.bar(np.arange(start, stop), avg_dist_by_k)
    plt.xlabel('k')
    plt.ylabel('Avg distance from cluster center')
    plt.savefig(path + '/inertia.png')
    plt.clf()


def save_test_results(imgs_dict, acts_dict, centers, path):
    # make the directory to store these files
    try:
        os.mkdir(path)
    except OSError:
        print('Creation of the directory %s failed' % path)
    else:
        print('Successfully created the directory %s ' % path)

    # Save centers
    center_data_file = open(path + 'data_centers.txt', 'w')
    for i, center in enumerate(centers):
        center_data_file.write(str(i) + ':' + str(center) + '\n')
    center_data_file.close()

    num_matches = 100
    avg_cluster_imgs = []
    top_matches_data_file = open(path + 'data_top_matches.txt', 'w')
    for i, center in enumerate(centers):
        print('Saving results for cluster ' + str(i))
        matches_acts = acts_dict[i]
        matches_img = imgs_dict[i]
        target = center

        # get the error for all the acts matching this cluster and sort
        error = []
        for acts_piece in matches_acts:
            error.append(np.sum((acts_piece - target) ** 2))
        sort_idx = np.argsort(error)

        # get the top matches
        top_matches = []
        top_matches_ids = []
        num = min(num_matches, len(matches_img))
        top_matches_data_file.write('CENTER' + str(i) + '\n')
        for idx in range(num):
            match_index = sort_idx[idx]
            top_match = matches_img[match_index]
            top_matches.append(top_match)
            top_matches_ids.append(idx)
            top_match_acts = matches_acts[match_index]
            top_matches_data_file.write(str(idx) + ':' + str(top_match_acts) + '\n')

        # make average image of top matches
        avg_cluster_img = make_avg_img(top_matches)
        avg_cluster_imgs.append(avg_cluster_img)

        save_imgs(top_matches, path + 'top_matches_' + ('0' if i < 10 else '') + ('00' if i < 100 else '') + str(i), top_matches_ids)

    center_data_file.close()
    save_imgs(avg_cluster_imgs, path + 'top_matches_avg')
    print('Finished saving')


def test_k(imgs, acts, k_size, path):
    # run kmeans
    print('Start K-means with k=' + str(k_size))
    km = KMeans(n_clusters=k_size, init='k-means++')
    assignments = km.fit_predict(acts)
    centers = km.cluster_centers_
    print('Finished K-means')

    # get pure versions of the centers (ie. only preserve outliers)
    # pure_centers = []
    # visual_concepts = []
    # for center in centers:
    #     outlier_idx = get_outliers(center)
    #     pure_center = []
    #     for i, item in enumerate(center):
    #         if i in outlier_idx:
    #             pure_center.append(item)
    #         else:
    #             pure_center.append(0)
    #     visual_concepts.append(outlier_idx)
    #     pure_centers.append(pure_center)

    # make the directory to store these files
    # try:
    #     os.mkdir(path)
    # except OSError:
    #     print('Creation of the directory %s failed' % path)
    # else:
    #     print('Successfully created the directory %s ' % path)

    # visual_concepts_file = open(path + 'visual_concepts.txt', 'w')
    # for idx in visual_concepts:
    #     visual_concepts_file.write(str(idx) + '\n')
    # visual_concepts_file.write('\nCenters\n')
    # for center in centers:
    #     visual_concepts_file.write(str(center) + '\n')
    # visual_concepts_file.write('\nPure Centers\n')
    # for center in pure_centers:
    #     visual_concepts_file.write(str(center) + '\n')
    # visual_concepts_file.close()

    # sort the assignments into buckets
    print('Sorting results')
    acts_dict = {}
    imgs_dict = {}
    for i, assignment in enumerate(assignments):
        img = imgs[i]
        act = acts[i]
        if assignment not in acts_dict:
            acts_dict[assignment] = []
            imgs_dict[assignment] = []
        acts_dict[assignment].append(act)
        imgs_dict[assignment].append(img)

    # save bar charts with stats and images for visualizing clusters
    save_test_results(imgs_dict, acts_dict, centers, path)


# # steps = [10, 1, 100, 100, 100]
# pcts = [0.01, 0.4, 0, 0, 0]
# thresholds = [0.03, 0.03, 0.03, 0.03, 0.03]
# thresholdPcts = [0.1, 0.5, 0.03, 0.03, 0.03]
# load images chopped up into pieces with acts for each at the layer


def get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct):
    acts_pieces_by_layer, img_pieces_by_layer, layers, layer_names = load_corpus(sample_rate, [layer_name], [pct], [threshold], [thresholdPct])
    imgs = img_pieces_by_layer[0]
    acts = acts_pieces_by_layer[0]
    return imgs, acts


def layer1():
    layer_name = 'conv1'
    sample_rate = 1200
    pct = 0.01
    threshold = 0.03
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)

    # for k_size in [25, 26, 27, 28, 29]:
    #     ts = str(int(time.time()))[3:]
    #     print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
    #     path = '../results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
    #     # path = '../results/' + layer_name + '/find__t' + ts + '_n' + str(len(imgs)) + '/'
    #     test_k(imgs, acts, k_size, path)

    # should be in the range (18, 32) based on find_k
    k_range = (1, 2)
    ts = str(int(time.time()))[3:]
    path = '../results/find_k/conv1_t' + ts + '_n' + str(len(imgs)) + '/'
    print('Find K with n=' + str(len(imgs)) + ' and f=' + str(len(acts)))
    find_k(imgs, acts, k_range, path)


def layer2():
    # params for L2
    layer_name = 'conv2'
    sample_rate = 400
    pct = 0.02
    threshold = 0.03
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)

    for k_size in [20]:
        ts = str(int(time.time()))[3:]
        print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
        path = '../results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
        test_k(imgs, acts, k_size, path)

    # k_range = (1, 2)
    # ts = str(int(time.time()))[3:]
    # path = '../results/find_k/conv2_t' + ts + '_n' + str(len(imgs)) + '/'
    # print('Find K with n=' + str(len(imgs)) + ' and f=' + str(len(acts)))
    # find_k(imgs, acts, k_range, path)


def layer3():
    # params for L3
    layer_name = 'conv3'
    sample_rate = 20
    pct = 0.4
    threshold = 0.03
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)

    for k_size in [128, 212, 300]:
    # for k_size in [5]:
        ts = str(int(time.time()))[3:]
        print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
        path = '../results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
        test_k(imgs, acts, k_size, path)

    # k_range = (1, 513)
    # ts = str(int(time.time()))[3:]
    # path = '../results/find_k/conv3_t' + ts + '_n' + str(len(imgs)) + '/'
    # print('Find K with n=' + str(len(imgs)) + ' and f=' + str(len(acts)))
    # find_k(imgs, acts, k_range, path)


def layer4():
    # params for L4
    layer_name = 'conv4'
    sample_rate = 1200
    pct = 0.05
    threshold = 0.03
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)

    # for k_size in [24, 24, 36, 36, 48, 48, 60, 60, 72, 72, 84, 84, 96, 96]:
    #     ts = str(int(time.time()))[3:]
    #     print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
    #     path = '../results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
    #     # path = '../results/' + layer_name + '/find__t' + ts + '_n' + str(len(imgs)) + '/'

    #     test_k(imgs, acts, k_size, path)

    k_range = (1, 2)
    ts = str(int(time.time()))[3:]
    path = '../results/find_k/conv4_t' + ts + '_n' + str(len(imgs)) + '/'
    print('Find K with n=' + str(len(imgs)) + ' and f=' + str(len(acts)))
    find_k(imgs, acts, k_range, path)


def layer5():
    # params for L5
    layer_name = 'conv5'
    sample_rate = 1200
    pct = 0.1
    threshold = 0.03
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)

    # for k_size in [128, 84, 72, 96, 96]:
    #     ts = str(int(time.time()))[3:]
    #     print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
    #     path = '../results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
    #     # path = '../results/' + layer_name + '/find__t' + ts + '_n' + str(len(imgs)) + '/'

    #     test_k(imgs, acts, k_size, path)

    k_range = (1, 2)
    ts = str(int(time.time()))[3:]
    path = '../results/find_k/conv5_t' + ts + '_n' + str(len(imgs)) + '/'
    print('Find K with n=' + str(len(imgs)) + ' and f=' + str(len(acts)))
    find_k(imgs, acts, k_range, path)


if __name__ == '__main__':
    # layer1()
    layer2()
    # layer3()
    # layer4()
    # layer5()
