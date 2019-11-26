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


# np.set_printoptions(threshold=np.inf)

# extract the centers from a certain type of file
def get_centers(from_path, idxs):
    # open from path
    from_file = open(from_path)
    # read
    text = from_file.read()
    centers_text = text.split('Pure Centers')[0].split('Centers')[1]

    parts = centers_text.split(']\n[')

    centers = []
    for idx in idxs:
        nums = parts[idx].split(',')
        nums = map(float, nums)
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


def make_avg_img_align(imgs, alignType=None, keepNoAlign=True):
    print('make_avg_img')
    if len(imgs) == 0:
        return None

    # get start img
    # avg_cluster_img = imgs[0].copy().squeeze()
    total_img = imgs[0].copy().squeeze()
    total_img_norm = imgs[0].copy().squeeze()
    for i, img in enumerate(imgs[1:]):
        total_img_norm = cv2.normalize(total_img, total_img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        avg_cluster_img_8U = np.uint8(total_img_norm).squeeze()
        img_8U = np.uint8(img * 255).squeeze()
        # try to align img with avg
        img_aligned = img_8U

        if alignType is not None:
            try:
                img_aligned = alignImages(avg_cluster_img_8U, img_8U, alignType)
                if not keepNoAlign:
                    total_img += img_aligned / 255.0
                print('align')
            except cv2.error as e:
                print('no align')
                # cannot align
                pass

        # take average of current and new image (weighted so that each image is averaged equally)
        # avg_cluster_img += img_aligned * (1.0 / (i + 1)) / 255
        # cv2.normalize(avg_cluster_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if keepNoAlign:
            total_img += img_aligned / 255.0

    total_img = cv2.normalize(total_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return total_img


def make_avg_img(imgs, alignType=None):
    print('make_avg_img')
    if len(imgs) == 0:
        return None

    # get start img
    # avg_cluster_img = imgs[0].copy().squeeze()
    total_img = imgs[0].copy()
    for i, img in enumerate(imgs[1:]):
        total_img += img

    total_img = cv2.normalize(total_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return total_img


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
    # number_of_iterations = 5000
    number_of_iterations = 50

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    # termination_eps = 1e-10
    termination_eps = 0.001

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
        # print('target', center)

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
        avg_img_align0a = make_avg_img_align(top_matches, alignType=cv2.MOTION_EUCLIDEAN, keepNoAlign=False)
        avg_img_align0b = make_avg_img_align(top_matches, alignType=cv2.MOTION_EUCLIDEAN, keepNoAlign=True)
        avg_imgs_by_center.append([avg_img, avg_img_align0a, avg_img_align0b])
        print('Finished making avg images')

        print('Start saving images')
        save_imgs([avg_img], path + 'top_matches_avg_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i), num_cols=1, pad=0)
        save_imgs([avg_img_align0a], path + 'top_matches_avg_align_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i), num_cols=1, pad=0)
        save_imgs([avg_img_align0b], path + 'top_matches_avg_align_only_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i), num_cols=1, pad=0)
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
        item = np.array2string(center, separator=',')
        centers_data_file.write(item + '\n')

    # save top match avg and aligned avg images pngs
    save_imgs([i[2] for i in avg_imgs_by_center], path + 'top_matches_avg_align')
    save_imgs([i[1] for i in avg_imgs_by_center], path + 'top_matches_avg_align_only')
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
    # print(text[:10], text[-10:])
    text = text[1:-2]
    # print(text[:10], text[-10:])
    parts = text.split(']\n[')
    centers = []
    for idx in idxs:
        nums = parts[idx].split(',')
        nums = map(float, nums)
        centers.append(nums)

    return centers


def get_top_matches_data(from_path, idxs):
    # open from path
    from_file = open(from_path)
    # read
    text = from_file.read()
    matches_text_by_center = text.split(':')

    matches_by_center = []
    for idx in idxs:
        part_text = matches_text_by_center[idx]
        # print(part_text[:10], part_text[-10:])
        part_text = part_text[1:-2]
        # print(part_text[:10], part_text[-10:])
        parts = part_text.split(']\n[')
        matches = []
        for part in parts:
            nums = part.split(',')
            nums = [n.strip() for n in nums]
            # print(nums)
            nums = map(float, nums)
            matches.append(nums)
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
    # L2
    # center_allstars = [
    #     ['cluster_t3842748_k=100_n=11762', [1, 20, 89]],
    #     ['cluster_t3842548_k=100_n=11719', [64]],
    #     ['cluster_t3842342_k=100_n=11981', [76, 83, 37, 47, 31]],

    #     ['cluster_t3842144_k=80_n=11836', []],
    #     ['cluster_t3841947_k=80_n=11686', []],
    #     ['cluster_t3841749_k=80_n=11761', [7, 69]],

    #     ['cluster_t3841550_k=80_n=11723', [36, 12, 41, 49, 2, 78, 65]],
    #     ['cluster_t3841344_k=80_n=11541', [66, 34]],
    #     ['cluster_t3841126_k=80_n=11759', [46]],

    #     ['cluster_t3840901_k=80_n=11945', [55, 67, 73, 23, 0, 78, 38]],
    #     ['cluster_t3840679_k=80_n=11972', [0, 6, 17]],
    # ]

    # # L3
    # center_allstars = [
    #     ['cluster_t3858236_k=120_n=14335', [50, 60, 5, 37, 74, 67, 85, 93, 33, 63, 108, 4]],
    #     ['cluster_t3857752_k=120_n=14363', [0, 67, 81, 22, 44, 49, 59, 73, 79, 104, 115]],
    #     ['cluster_t3857260_k=120_n=14496', [1, 7, 14, 20, 49, 52, 79, 83, 87, 90, 101, 112, 71]],

    #     ['cluster_t3856783_k=100_n=14400', [34, 1, 2, 3, 8, 9, 22, 23, 26, 28, 41, 30, 43, 32]],
    #     ['cluster_t3856306_k=100_n=14450', [73, 16]],
    #     ['cluster_t3856783_k=100_n=14400', [44]],
    # ]

    # # L4
    # center_allstars = [
    #     ['cluster_t3867102_k=160_n=21793', [2, 3, 4, 10, 11, 13, 14, 16, 18, 19, 20, 21, 23, 26, 27, 28, 29, 34, 38, 40, 45, 46, 48, 50, 52, 54, 57, 61, 62, 63, 65, 67, 68, 73, 81, 82, 83, 93, 97, 100, 102, 107, 110, 120, 121, 123, 136, 137, 138, 139, 145, 146, 148, 154]],
    #     ['cluster_t3866309_k=160_n=21771', [159]],
    #     ['cluster_t3865557_k=120_n=21818', [57, 43]],
    # ]

    # L5
    center_allstars = [
        ['cluster_t3879642_k=300_n=28749', [20, 32]],
        ['cluster_t3877401_k=200_n=28883', [46, 54, 55, 145, 176, 163, 192]],  # legs
        ['cluster_t3879642_k=300_n=28749', [35, 22]],
        ['cluster_t3877401_k=200_n=28883', [6, 115, 134, 193]],  # other body parts
        ['cluster_t3879642_k=300_n=28749', [23, 27, 43, 57, 65, 66, 67]],
        ['cluster_t3877401_k=200_n=28883', [36, 41, 57, 59, 68, 120, 125, 178, 181]],  # pipes
        ['cluster_t3879642_k=300_n=28749', [36, 39, 40, 56, 58, 68, 69, 70]],
        ['cluster_t3877401_k=200_n=28883', [28, 64, 93, 189]],  # corners
        ['cluster_t3879642_k=300_n=28749', [28]],
        ['cluster_t3877401_k=200_n=28883', [37, 39, 44, 197, 199]],  # tips
        ['cluster_t3879642_k=300_n=28749', []],
        ['cluster_t3877401_k=200_n=28883', [24, 29, 53, 77, 80, 87, 102, 119, 100, 163, 196]], # fields
    ]

    # Collect centers and top match data into files
    all_centers = []
    all_top_match_data = []
    all_avg_imgs = []
    all_top_img_paths = []
    print('Collect data')
    for row in center_allstars:
        print(row[0])
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
        to_file.write(','.join([str(f) for f in center]) + '\n')

    to_file = open(path + 'top_matches_data.txt', 'w')
    for i, matches in enumerate(all_top_match_data):
        for match in matches:
            to_file.write(','.join([str(f) for f in match]) + '\n')
        to_file.write(':\n')
    print('all_top_match_data', len(all_top_match_data))

    for i, old_path in enumerate(all_top_img_paths):
        new_path = path + 'top_matches_' + ('0' if i < 10 else '') + ('0' if i < 100 else '') + str(i) + '.png'
        shutil.copyfile(old_path, new_path)

    save_imgs([i[1] for i in all_avg_imgs], path + 'top_matches_avg')
    save_img_groups(all_avg_imgs, path + 'top_matches_avg_comp')


def load_layer2():
    # params for L2
    layer_name = 'conv2'
    sample_rate = 25
    pct = 0.02
    threshold = 0.03
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)
    return imgs, acts


def load_layer3():
    # params for L3
    layer_name = 'conv3'
    sample_rate = 20
    pct = 0.07
    threshold = 0.05
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)
    return imgs, acts


def load_layer4():
    # params for L4
    layer_name = 'conv4'
    sample_rate = 20
    pct = 0.1
    threshold = 0.1
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)
    return imgs, acts


def load_layer5():
    # params for L5
    layer_name = 'conv5'
    sample_rate = 20
    pct = 0.13
    threshold = 0.1
    thresholdPct = 0.1

    imgs, acts = get_imgs_and_acts(layer_name, sample_rate, pct, threshold, thresholdPct)
    return imgs, acts


if __name__ == '__main__':
    # path = '../docs/data/conv2/'
    # old_path = '../results/conv2/'
    # path = '../docs/data/conv3/'
    # old_path = '../results/conv3/'
    # path = '../docs/data/conv4/'
    # old_path = '../results/conv4/'
    path = '../docs/data/conv5/'
    old_path = '../results/conv5/'
    remove_old_files(path)
    finalize(path, old_path)


    # centers = combine_centers2(old_path)

    # for k in [80, 80, 80, 80, 80, 80, 80, 80, 100, 100, 100]:
    # # for k in [10]:
    #     ts = str(int(time.time()))[3:]
    #     imgs, acts = load_layer2()
    #     path = '../results/conv2/cluster_t' + ts + '_k=' + str(k) + '_n=' + str(len(acts)) + '/'

    #     try:
    #         os.mkdir(path)
    #     except OSError:
    #         print('Creation of the directory %s failed' % path)
    #     else:
    #         print('Successfully created the directory %s ' % path)

    #     centers = []
    #     generate_data(centers, imgs, acts, path, k, num_matches=100)


    # for k in [80, 80, 80, 100, 100, 100, 120, 120, 120, 160, 200, 240, 300]:
    # # for k in [10]:
    #     ts = str(int(time.time()))[3:]
    #     imgs, acts = load_layer3()
    #     path = '../results/conv3/cluster_t' + ts + '_k=' + str(k) + '_n=' + str(len(acts)) + '/'

    #     try:
    #         os.mkdir(path)
    #     except OSError:
    #         print('Creation of the directory %s failed' % path)
    #     else:
    #         print('Successfully created the directory %s ' % path)

    #     centers = []
    #     generate_data(centers, imgs, acts, path, k, num_matches=100)


    # for k in [80, 100, 100, 120, 120, 160, 160, 200, 240, 300]:
    # # for k in [10]:
    #     ts = str(int(time.time()))[3:]
    #     imgs, acts = load_layer4()
    #     path = '../results/conv4/cluster_t' + ts + '_k=' + str(k) + '_n=' + str(len(acts)) + '/'

    #     try:
    #         os.mkdir(path)
    #     except OSError:
    #         print('Creation of the directory %s failed' % path)
    #     else:
    #         print('Successfully created the directory %s ' % path)

    #     centers = []
    #     generate_data(centers, imgs, acts, path, k, num_matches=100)


    # for k in [80, 100, 100, 120, 120, 160, 160, 200, 240, 300]:
    # # for k in [10]:
    #     ts = str(int(time.time()))[3:]
    #     imgs, acts = load_layer5()
    #     path = '../results/conv5/cluster_t' + ts + '_k=' + str(k) + '_n=' + str(len(acts)) + '/'

    #     try:
    #         os.mkdir(path)
    #     except OSError:
    #         print('Creation of the directory %s failed' % path)
    #     else:
    #         print('Successfully created the directory %s ' % path)

    #     centers = []
    #     generate_data(centers, imgs, acts, path, k, num_matches=100)




