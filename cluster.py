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

from sketch_a_net import load_layers, load_model, get_avg_pools, load_pretrained


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
    if len(imgs) <= 20:
        row_size = 4.0
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

    avg_cluster_img = imgs[0].squeeze()
    for i, img in enumerate(imgs[1:]):
        avg_cluster_img_8U = np.uint8(avg_cluster_img * 255).squeeze()
        img_8U = np.uint8(img * 255).squeeze()
        # try to align img with avg
        img_aligned = img_8U
        try:
            img_aligned = alignImages(avg_cluster_img_8U, img_8U)
        except cv2.error as e:
            # cannot align
            pass

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


def get_imgs_and_acts(layer_name, sample_rate, step, output_size, stride, f_size, padding=0):
    # load images from dataset
    data = h5py.File('./dataset_without_order_info_224.mat', 'r')
    all_imgs = data['imdb']['images']['data']
    all_labels = data['imdb']['images']['labels']

    # reduce size
    # imgs = [all_imgs[0]]
    # labels = [all_labels[0]]
    imgs = []
    labels = []
    for i in range(0, len(all_imgs), sample_rate):
        imgs.append(all_imgs[i])
        labels.append(all_labels[i])
    print('Chose ' + str(len(imgs)) + ' images')

    # format images for Sketch-A-Net
    imgs_f = []
    for img in imgs:
        # resize and format image
        img = img.swapaxes(0, 2)
        img = img[16:241, 16:241, :]
        img = img / 255
        img = 1 - img
        imgs_f.append(img)
    imgs = imgs_f

    # load layers of model
    layers = load_layers('./model_without_order_info_224.mat')

    # get activations for each image at given layer
    acts_by_img = []
    print('Calculating activations for images(' + str(len(imgs)) + ')')
    for img in imgs:
        sys.stdout.write('.'),
        acts = get_layer_output(layers, layer_name, img)
        acts_by_img.append(acts)
    print('Finished getting activations')

    # divide activations and images into sections
    print('Dividing images(' + str(len(imgs)) + ') into cropped sections')

    section_acts_list = []
    section_img_list = []
    count = 0
    for i, img in enumerate(imgs):
        sys.stdout.write('.'),
        acts = acts_by_img[i]

        if padding > 0:
            img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # # filter out those that are close to blank, since many are
        thresh = (f_size ** 2) * 0.01
        # but keep some
        portion_to_keep_below_thresh = 0.2
        for y in range(0, output_size, step):
            for x in range(0, output_size, step):
                count += 1
                # get section of original image
                x_start = x * stride
                x_end = x_start + f_size
                y_start = y * stride
                y_end = y_start + f_size

                section_img = img[y_start: y_end, x_start: x_end]

                # ignore blank images
                if np.sum(section_img) < thresh and random.random() > portion_to_keep_below_thresh:
                    continue

                section_img_list.append(section_img)

                # get activation at point
                section_acts = acts[0, y, x, :]

                # L2 normalization
                feat_norm = np.sqrt(np.sum(section_acts ** 2))
                if feat_norm > 0:
                    section_acts = section_acts / feat_norm
                section_acts_list.append(section_acts)

    # check that they match
    print('These should match: acts(' + str(len(section_acts_list)) + ') == imgs(' + str(len(section_img_list)) + ') out of ' + str(count) + ' possible')

    return section_img_list, section_acts_list


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
        assignments = km.fit_predict(acts)
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


def test_k(imgs, acts, k_size, path):
    # run kmeans at different k's
    print('Start K-means with k=' + str(k_size))
    km = KMeans(n_clusters=k_size, init='k-means++')
    assignments = km.fit_predict(acts)
    centers = km.cluster_centers_
    print('Finished K-means')

    # get pure versions of the centers (ie. only preserve outliers)
    pure_centers = []
    visual_concepts = []
    for center in centers:
        outlier_idx = get_outliers(center)
        pure_center = []
        for i, item in enumerate(center):
            if i in outlier_idx:
                pure_center.append(item)
            else:
                pure_center.append(0)
        visual_concepts.append(outlier_idx)
        pure_centers.append(pure_center)

    # make the directory to store these files
    try:
        os.mkdir(path)
    except OSError:
        print('Creation of the directory %s failed' % path)
    else:
        print('Successfully created the directory %s ' % path)

    visual_concepts_file = open(path + 'visual_concepts.txt', 'w')
    for idx in visual_concepts:
        visual_concepts_file.write(str(idx) + '\n')
    visual_concepts_file.write('\nCenters\n')
    for center in centers:
        visual_concepts_file.write(str(center) + '\n')
    visual_concepts_file.write('\nPure Centers\n')
    for center in pure_centers:
        visual_concepts_file.write(str(center) + '\n')
    visual_concepts_file.close()

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
    save_results(imgs_dict, acts_dict, centers, path)


def layer1():
    # params for L1
    layer_name = 'conv1'
    output_size = 71
    stride = 3
    f_size = 15
    padding = 0
    # should be in the range (18, 32) based on find_k
    k_range = (1, 65)
    sample_rate = 320
    step = 4

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, step, output_size, stride, f_size, padding)

    for k_size in [25, 26, 27, 28, 29]:
        ts = str(int(time.time()))[3:]
        print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
        path = './results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
        # path = './results/' + layer_name + '/find__t' + ts + '_n' + str(len(imgs)) + '/'

        test_k(imgs, acts, k_size, path)
        # find_k(imgs, acts, k_range, path)

def layer2():
    # params for L2
    layer_name = 'conv2'
    output_size = 31
    # in terms of original image
    stride = 6
    f_size = 45
    padding = 0
    # should be in the range () based on find_k
    # k_range = (1, 65)
    sample_rate = 320
    step = 2

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, step, output_size, stride, f_size, padding)

    for k_size in [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]:
        ts = str(int(time.time()))[3:]
        print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
        path = './results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
        # path = './results/' + layer_name + '/find__t' + ts + '_n' + str(len(imgs)) + '/'

        test_k(imgs, acts, k_size, path)
        # find_k(imgs, acts, k_range, path)


def layer3():
    # params for L3
    layer_name = 'conv3'
    output_size = 15
    # in terms of original image
    stride = 12
    f_size = 81
    padding = 12
    # should be in the range () based on find_k
    # k_range = (1, 65)
    sample_rate = 320
    step = 1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, step, output_size, stride, f_size, padding)

    for k_size in [24, 24, 36, 36, 48, 48, 60, 60, 72, 72, 84, 84, 96, 96]:
        ts = str(int(time.time()))[3:]
        print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
        path = './results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
        # path = './results/' + layer_name + '/find__t' + ts + '_n' + str(len(imgs)) + '/'

        test_k(imgs, acts, k_size, path)
        # find_k(imgs, acts, k_range, path)


def layer4():
    # params for L4
    layer_name = 'conv4'
    output_size = 15
    # in terms of original image
    stride = 12
    f_size = 105
    padding = 24
    # should be in the range () based on find_k
    # k_range = (1, 65)
    sample_rate = 320
    step = 1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, step, output_size, stride, f_size, padding)

    for k_size in [24, 24, 36, 36, 48, 48, 60, 60, 72, 72, 84, 84, 96, 96]:
        ts = str(int(time.time()))[3:]
        print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
        path = './results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
        # path = './results/' + layer_name + '/find__t' + ts + '_n' + str(len(imgs)) + '/'

        test_k(imgs, acts, k_size, path)
        # find_k(imgs, acts, k_range, path)


def layer5():
    # params for L5
    layer_name = 'conv5'
    output_size = 15
    # in terms of original image
    stride = 12
    f_size = 129
    padding = 36
    # should be in the range () based on find_k
    k_range = (1, 65)
    sample_rate = 320
    step = 1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, step, output_size, stride, f_size, padding)

    for k_size in [128, 84, 72, 96, 96]:
        ts = str(int(time.time()))[3:]
        print('With n=' + str(len(imgs)) + ' and f=' + str(len(acts[0])))
        path = './results/' + layer_name + '/cluster_t' + ts + '_k' + str(k_size) + '_n' + str(len(imgs)) + '/'
        # path = './results/' + layer_name + '/find__t' + ts + '_n' + str(len(imgs)) + '/'

        test_k(imgs, acts, k_size, path)
        # find_k(imgs, acts, k_range, path)


if __name__ == '__main__':
    layer2()
