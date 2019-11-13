import numpy as np
import h5py
from sklearn.cluster import KMeans
import cv2
import math
import time
import sys
import os
import random
import re
import glob
import shutil
from data_api import load_corpus


# extract the centers from a certain type of file
def get_centers(from_path, idxs):
    # open from path
    from_file = open(from_path)
    # read
    text = from_file.read()
    centers_text = text.split('Pure Centers')[0].split('Centers')[1]

    parts = centers_text.split(']\n[')

    centers = []
    for i, part in enumerate(parts):
        match = re.findall('\d+\.\d+', part)
        if i in idxs:
            centers.append(match)

    return centers


def read_centers(path):
    file_to_read = open(path)
    text = file_to_read.read()
    lines = text.split('\n')
    result = []
    for line in lines:
        if line != '':
            values = line.split(',')
            result.append([float(v) for v in values])
    return result


def get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct):
    acts_pieces_by_layer, img_pieces_by_layer, layers, layer_names = load_corpus(sample_rate, [layer_name], [pct], [threshold], [thresholdPct])
    imgs = img_pieces_by_layer[0]
    acts = acts_pieces_by_layer[0]
    return imgs, acts


def make_avg_img(imgs, alignType=None):
    if len(imgs) == 0:
        return None

    # get start img
    avg_cluster_img = imgs[0].copy().squeeze()

    for i, img in enumerate(imgs[1:]):
        avg_cluster_img_8U = np.uint8(avg_cluster_img * 255).squeeze()
        img_8U = np.uint8(img * 255).squeeze()
        # try to align img with avg
        img_aligned = img_8U

        if alignType is not None:
            try:
                img_aligned = alignImages(avg_cluster_img_8U, img_8U, alignType)
            except cv2.error as e:
                # cannot align
                pass

        # take average of current and new image (weighted so that each image is averaged equally)
        avg_cluster_img += img_aligned * (1.0 / (i + 1)) / 255
        cv2.normalize(avg_cluster_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return avg_cluster_img


def alignImages(im1_gray, im2_gray, alignType):
    '''
    Modified From: https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    '''
    # Find size of image1
    sz = im1_gray.shape

    # Define the motion model
    # warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = alignType

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


def save_img_groups(img_groups, name):
    last = None
    for g in img_groups:
        for img in g:
            if img is not None:
                last = img
    if last is None:
        return
    h, w = last.shape

    # Make cv img and write to it...
    pad = 2
    num_rows = len(img_groups)
    num_cols = len(img_groups[0])
    total_img = np.ones(((h + (2 * pad)) * num_rows, (w + (2 * pad)) * num_cols))

    for i, group in enumerate(img_groups):
        for j, img in enumerate(group):
            row = i
            col = j
            if img is not None:
                img_pad = img.squeeze()
                h, w = img_pad.shape
                y = ((h + (pad * 2)) * row) + pad
                y_end = y + h
                x = ((w + (pad * 2)) * col) + pad
                x_end = x + w
                total_img[y:y_end, x:x_end] = img_pad

    cv2.imwrite(name + '.png', total_img * 255)


def save_imgs(imgs, name, num_cols=10, pad=2):
    if len(imgs) == 0 or imgs[0] is None:
        return

    h = imgs[0].shape[0]
    w = imgs[0].shape[1]

    # Make cv img and write to it...
    num_rows = int(math.ceil(len(imgs) / float(num_cols)))
    total_img = np.ones(((h + (2 * pad)) * num_rows, (w + (2 * pad)) * num_cols))

    print('Start compile image')
    for i, img in enumerate(imgs):
        row = int(math.floor(i / float(num_cols)))
        col = i % num_cols
        if img is not None:
            img_pad = img.squeeze()
            h, w = img_pad.shape
            y = ((h + (pad * 2)) * row) + pad
            y_end = y + h
            x = ((w + (pad * 2)) * col) + pad
            x_end = x + w
            total_img[y:y_end, x:x_end] = img_pad
    print('Finished compile image')

    print('Start saving image')
    cv2.imwrite(name + '.png', total_img * 255)
    print('Finished saving image')


def generate_data(centers, imgs, acts, path, k, num_matches=100):
    # pad the data with known centers?
    # original_len = len(acts)
    # print('Original length: ' + str(original_len))
    # h, w, c = imgs[0].shape
    # placeholder = np.ones((h, w, c))
    # for center in centers:
    #     # pad = len(acts) / k
    #     # pad = len(acts) / k
    #     # pad = 20
    #     pad = 1
    #     for i in range(pad):
    #         acts.append(center)
    #         imgs.append(placeholder)

    # Cluster acts with kmeans
    print('Start K-means with known centers')
    # km = KMeans(n_clusters=128, n_init=1, init=np.array(centers))
    km = KMeans(n_clusters=k, init='k-means++')
    assignments = km.fit_predict(acts)
    # update centers
    centers = km.cluster_centers_
    # assignments = km.predict(acts)

    # print('Finished K-means')
    # sort the assignments into buckets
    print('Sorting results')
    acts_dict = {}
    imgs_dict = {}
    for i in range(len(centers)):
        acts_dict[i] = []
        imgs_dict[i] = []

    # prune assignments to real data points
    # assignments = assignments[:original_len]
    print('Assign length: ' + str(len(assignments)))
    for i, assignment in enumerate(assignments):
        acts_dict[assignment].append(acts[i])
        imgs_dict[assignment].append(imgs[i])
    print('Done sorting results')

    # record all avg imgs
    avg_imgs_by_center = []
    # record all top_matches data
    top_matches_acts_by_center = []
    for i, center in enumerate(centers):
        print('Saving results for cluster ' + str(i))
        matches_acts = acts_dict[i]
        matches_img = imgs_dict[i]
        target = center

        # get the error for all the acts matching this cluster and sort
        print('Start calculating error')
        error = []
        for acts_piece in matches_acts:
            error.append(np.sum((acts_piece - target) ** 2))
        print('Finished calculating error')
        sort_idx = np.argsort(error)

        # print the image part matches
        top_matches = []
        top_matches_ids = []
        top_matches_acts = []
        for idx in range(min(num_matches, len(matches_img))):
            match_index = sort_idx[idx]
            top_match = matches_img[match_index]
            top_matches.append(top_match)
            top_match_acts = matches_acts[match_index]
            top_matches_acts.append(top_match_acts)
            top_matches_ids.append(idx)
        top_matches_acts_by_center.append(top_matches_acts)

        # generate avg img from top 100 and 5 aligned avg imgs
        print('Start making avg images')
        # avg_img = make_avg_img(top_matches)
        # avg_img_align0 = make_avg_img(top_matches, alignType=cv2.MOTION_TRANSLATION)
        # avg_img_align1 = make_avg_img(top_matches, alignType=cv2.MOTION_EUCLIDEAN)
        # avg_img_align2 = make_avg_img(top_matches, alignType=cv2.MOTION_AFFINE)
        # avg_img_align3 = make_avg_img(top_matches, alignType=cv2.MOTION_HOMOGRAPHY)
        # avg_imgs_by_center.append([avg_img, avg_img_align0, avg_img_align1, avg_img_align2, avg_img_align3])

        avg_img = make_avg_img(top_matches)
        avg_img_align0 = make_avg_img(top_matches, alignType=cv2.MOTION_EUCLIDEAN)
        avg_imgs_by_center.append([avg_img, avg_img_align0])
        print('Finished making avg images')

        print('Start saving images')
        save_imgs([avg_img], path + 'top_matches_avg_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i), num_cols=1, pad=0)
        save_imgs([avg_img_align0], path + 'top_matches_avg_align_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i), num_cols=1, pad=0)
        # Save the top 100 img for this cluster
        save_imgs(top_matches, path + 'top_matches_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i))
        print('Finished saving images')

    # save top 100 datas for each center to text file
    top_matches_data_file = open(path + 'top_matches_data.txt', 'w')
    for group in top_matches_acts_by_center:
        for match in group:
            item = np.array2string(match.numpy(), separator=',')
            top_matches_data_file.write(item + '\n')
        top_matches_data_file.write(':')  # mark ends

    # save centers to text file
    centers_data_file = open(path + 'centers_data.txt', 'w')
    for center in centers:
        item = np.array2string(match.numpy(), separator=',')
        centers_data_file.write(item + '\n')

    # save top match avg and aligned avg images pngs
    save_imgs([i[1] for i in avg_imgs_by_center], path + 'top_matches_avg')
    save_img_groups(avg_imgs_by_center, path + 'top_matches_avg_comp')


# combine the best centers from a variety of runs
def combine_centers2(path):
    all_centers = []
    # from each run, find indices that you like and add to centers
    # centers = get_centers('../results/conv2/cluster_t2846945_k60_n11325/visual_concepts.txt', [i for i in range(60)])
    # all_centers.extend(centers)
    # centers = get_centers('../results/conv2/cluster_t2846043_k60_n11325/visual_concepts.txt', [2, 3])
    # all_centers.extend(centers)

    center_allstars = [
        ['cluster_t2845132_k60_n11325', [20, 30, 8, 4, 53, 48, 10]],
        ['cluster_t2844232_k60_n11325', [6, 47, 58]],
        ['cluster_t2843350_k60_n11325', [11, 8]],
        ['cluster_t2842460_k60_n11325', [57, 45, 1, 4]],
        ['cluster_t2841570_k60_n11325', [48, 10]],
        ['cluster_t2841570_k60_n11325', [19, 56, 2, 49, 26, 12, 39, 25, 45, 28, 4, 13]],
    ]

    for row in center_allstars:
        centers = get_centers(path + row[0] + '/visual_concepts.txt', row[1])

        centers_norm = []
        for center in centers:
            center = np.array(center).astype(float)
            feat_norm = np.sqrt(np.sum(center ** 2))
            if feat_norm > 0:
                center = center / feat_norm
            centers_norm.append(center)

        all_centers.extend(centers_norm)

    to_file = open(path + 'centers.txt', 'w')
    for center in all_centers:
        to_file.write(','.join(str(center)) + '\n')

    return all_centers


def load_layer2():
    # params for L2
    layer_name = 'conv2'
    sample_rate = 2000
    pct = 0.02
    threshold = 0.03
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)
    return imgs, acts


def remove_old_files(path):
    fileList = glob.glob(path + '*.png')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)


def get_centers2(from_path, idxs):
    # open from path
    from_file = open(from_path)
    # read
    text = from_file.read()
    parts = text.split(']\n[')
    centers = []
    for i, part in enumerate(parts):
        if i in idxs:
            match = re.findall('\d+\.\d*', part)
            centers.append(match)

    return centers


def get_top_matches_data(from_path, idxs):
    # open from path
    from_file = open(from_path)
    # read
    text = from_file.read()
    matches_text_by_center = text.split(':')

    matches_by_center = []
    for i, part_text in enumerate(matches_text_by_center):
        if i in idxs:
            parts = part_text.split(']\n[')
            matches = []
            for part in parts:
                match = re.findall('\d+\.\d*', part)
                matches.append(match)
            matches_by_center.append(matches)

    return matches_by_center


def get_center_avg_imgs(old_path, idxs):
    avg_imgs = []
    for i in idxs:
        img_path = old_path + 'top_matches_avg_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i) + '.png'
        avg_img = cv2.imread(img_path)
        avg_img = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)
        avg_img = avg_img / 255.0

        img_align_path = old_path + 'top_matches_avg_align_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i) + '.png'
        avg_img_align = cv2.imread(img_align_path)
        avg_img_align = cv2.cvtColor(avg_img_align, cv2.COLOR_BGR2GRAY)
        avg_img_align = avg_img_align / 255.0

        avg_imgs.append([avg_img, avg_img_align])

    return avg_imgs


def get_top_image_paths(old_path, idxs):
    paths = []
    for i in idxs:
        top_matches_img_path = old_path + 'top_matches_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i) + '.png'
        paths.append(top_matches_img_path)
    return paths


def finalize(path, old_path):
    # identify good centers from runs
    center_allstars = [
        ['cluster_t3676242_k=8_n=146', [0, 1]],
        ['cluster_t3676157_k=8_n=161', [3, 4]],
    ]

    # Collect centers and top match data into files
    all_centers = []
    all_top_match_data = []
    all_avg_imgs = []
    all_top_img_paths = []
    print('Collect data')
    for row in center_allstars:
        center_path = old_path + row[0] + '/'
        idxs = row[1]

        centers = get_centers2(center_path + 'centers_data.txt', idxs)
        all_centers.extend(centers)

        top_matches = get_top_matches_data(center_path + 'top_matches_data.txt', idxs)
        all_top_match_data.extend(top_matches)

        avg_imgs = get_center_avg_imgs(center_path, idxs)
        all_avg_imgs.extend(avg_imgs)

        # get top matches file and copy
        top_image_paths = get_top_image_paths(center_path, idxs)
        all_top_img_paths.extend(top_image_paths)

    print('Save data')
    to_file = open(path + 'centers_data.txt', 'w')
    for center in all_centers:
        to_file.write(','.join(center) + '\n')

    to_file = open(path + 'top_matches_data.txt', 'w')
    for matches in all_top_match_data:
        for match in matches:
            to_file.write(','.join(match) + '\n')
        to_file.write(':\n')

    for i, old_path in enumerate(all_top_img_paths):
        new_path = path + 'top_matches_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i) + '.png'
        print(new_path)
        shutil.copyfile(old_path, new_path)

    save_imgs([i[1] for i in all_avg_imgs], path + 'top_matches_avg')
    save_img_groups(all_avg_imgs, path + 'top_matches_avg_comp')


if __name__ == '__main__':
    path = '../docs/data/conv2/'
    old_path = '../results/conv2/'
    remove_old_files(path)
    finalize(path, old_path)
    # centers = combine_centers2(old_path)


    # centers = read_centers(path + 'centers.txt')
    # ts = str(int(time.time()))[3:]
    # k = 8
    # imgs, acts = load_layer2()
    # path = '../results/conv2/cluster_t' + ts + '_k=' + str(k) + '_n=' + str(len(acts)) + '/'

    # try:
    #     os.mkdir(path)
    # except OSError:
    #     print('Creation of the directory %s failed' % path)
    # else:
    #     print('Successfully created the directory %s ' % path)

    # centers = []
    # generate_data(centers, imgs, acts, path, k, num_matches=10)





