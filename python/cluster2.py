import numpy as np
import h5py
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import math
import time
import sys
import os
import random
import re
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


def make_avg_img(imgs, start_idx=0, align=False):
    if len(imgs) <= start_idx:
        return None

    # get start img
    avg_cluster_img = imgs[start_idx].copy().squeeze()

    for i, img in enumerate(imgs[1:]):
        avg_cluster_img_8U = np.uint8(avg_cluster_img * 255).squeeze()
        img_8U = np.uint8(img * 255).squeeze()
        # try to align img with avg
        img_aligned = img_8U

        if align:
            try:
                img_aligned = alignImages(avg_cluster_img_8U, img_8U)
            except cv2.error as e:
                # cannot align
                pass

        # take average of current and new image (weighted so that each image is averaged equally)
        avg_cluster_img += img_aligned * (1.0 / (i + 1)) / 255
        cv2.normalize(avg_cluster_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return avg_cluster_img


def alignImages(im1_gray, im2_gray):
    '''
    Modified From: https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    '''
    # Find size of image1
    sz = im1_gray.shape

    # Define the motion model
    # warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = cv2.MOTION_HOMOGRAPHY

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


def save_avg_imgs(img_groups, name):
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
    total_img = np.zeros(((h + (2 * pad)) * num_rows, (w + (2 * pad)) * num_cols))

    for i, group in enumerate(img_groups):
        for j, img in enumerate(group):
            if img is not None:
                img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                h, w = img_pad.shape
                y = h * i
                y_end = y + h
                x = w * j
                x_end = x + w
                total_img[y:y_end, x:x_end] = img_pad

    cv2.imwrite(name + '.png', total_img * 255)


def save_imgs(imgs, name):
    if len(imgs) == 0 or imgs[0] is None:
        return

    h, w, c = imgs[0].shape

    # Make cv img and write to it...
    pad = 2
    num_cols = 10
    num_rows = 10
    total_img = np.zeros(((h + (2 * pad)) * num_rows, (w + (2 * pad)) * num_cols))

    for i, img in enumerate(imgs):
        row = int(math.floor(i / float(num_cols)))
        col = i % num_cols
        if img is not None:
            img_pad = cv2.copyMakeBorder(img.squeeze(), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            h, w = img_pad.shape
            y = h * row
            y_end = y + h
            x = w * col
            x_end = x + w
            total_img[y:y_end, x:x_end] = img_pad

    cv2.imwrite(name + '.png', total_img * 255)


def generate_data(centers, imgs, acts, path, num_matches=2):
    # Cluster acts with kmeans
    print('Start K-means with known centers')
    km = KMeans(n_clusters=len(centers), n_init=1, init=np.array(centers))
    assignments = km.fit_predict(acts)
    # assignments = km.predict(acts)

    # print('Finished K-means')
    # sort the assignments into buckets
    print('Sorting results')
    acts_dict = {}
    imgs_dict = {}
    for i in range(len(centers)):
        acts_dict[i] = []
        imgs_dict[i] = []

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
        error = []
        for acts_piece in matches_acts:
            error.append(np.sum((acts_piece - target) ** 2))
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
        avg_img = make_avg_img(top_matches)
        avg_img_align0 = make_avg_img(top_matches, start_idx=0, align=True)
        avg_img_align1 = make_avg_img(top_matches, start_idx=1, align=True)
        avg_img_align2 = make_avg_img(top_matches, start_idx=2, align=True)
        avg_imgs_by_center.append([avg_img, avg_img_align0, avg_img_align1, avg_img_align2])

        # Save the top 100 img for this cluster
        save_imgs(top_matches, path + 'top_matches_' + ('0' if i < 10 else '') + ('00' if i < 100 else '') + str(i))

    # save top 100 datas for each center to text file
    top_matches_data_file = open(path + 'top_matches_data.txt', 'w')
    for group in top_matches_acts_by_center:
        for match in group:
            top_matches_data_file.write(str(match) + '\n')
        top_matches_data_file.write(':\n')  # mark ends

    # save top match avg and aligned avg images pngs
    save_avg_imgs(avg_imgs_by_center, path + 'top_matches_avg')


# combine the best centers from a variety of runs
def combine_centers2(path):
    all_centers = []
    # from each run, find indices that you like and add to centers
    centers = get_centers('../results/conv2/cluster_t2846945_k60_n11325/visual_concepts.txt', [i for i in range(60)])
    all_centers.extend(centers)
    # centers = get_centers('../results/conv2/cluster_t2846043_k60_n11325/visual_concepts.txt', [2, 3])
    # all_centers.extend(centers)

    to_file = open(path + 'centers.txt', 'w')
    for center in all_centers:
        to_file.write(','.join(center) + '\n')


def load_layer2():
    # params for L2
    layer_name = 'conv2'
    sample_rate = 40
    pct = 0.02
    threshold = 0.03
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)
    return imgs, acts


if __name__ == '__main__':
    path = '../docs/data/conv2/'
    combine_centers2(path)

    centers = read_centers(path + 'centers.txt')
    imgs, acts = load_layer2()
    generate_data(centers, imgs, acts, path)




