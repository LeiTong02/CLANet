import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import scipy.ndimage as ndi
from __future__ import (
    division,
    print_function,
)
import numpy as np
from tqdm import tqdm
import math
import os
from natsort import natsorted
from skimage.color import gray2rgb
from skimage.transform import resize
import skimage.data




def _generate_segments(im_orig, mask, request_bbox_size, bb_num_limit=500):

    # open the Image
    im_mask = skimage.measure.label(mask, return_num=True, background=0)
    # find top 100 region

    regions_pixels = np.bincount(im_mask[0].flatten())
    region_ids = np.argsort(regions_pixels)[::-1][:bb_num_limit]

    remove_mask = np.zeros((im_mask[0].shape[0], im_mask[0].shape[1]))

    for region_id in region_ids:
        remove_mask[im_mask[0] == region_id] = 1
    im_mask = im_mask[0] * remove_mask

    # im_mask  = im_mask[0]
    # merge mask channel to the image as a 4th channel
    im_orig = np.append(
        im_orig, np.zeros(im_orig.shape[:2])[:, :, np.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    return im_orig


def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_IoU(r1, r2):
    x_left = max(r1["min_x"], r2["min_x"])
    y_top = max(r1["min_y"], r2["min_y"])
    x_right = min(r1["max_x"], r2["max_x"])
    y_bottom = min(r1["max_y"], r2["max_y"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (r1["max_x"] - r1["min_x"]) * (r1["max_y"] - r1["min_y"])
    bb2_area = (r2["max_x"] - r2["min_x"]) * (r2["max_y"] - r2["min_y"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def _calc_sim(r1, r2, imsize):


    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_IoU(r1, r2))
    # return (_sim_IoU(r1,r2))


def _calc_colour_hist(img):
    """
        calculate colour histogram for each region
        the size of output histogram will be BINS * COLOUR_CHANNELS(3)
        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
        extract HSV
    """

    BINS = 25
    hist = np.array([])

    c = img[:, :, 0].flatten()

    hist = np.histogram(c, BINS, (0.0, 255.0))[0]

    #     for colour_channel in [0,]:

    #         # extracting one colour channel
    #         c = img[:,:, colour_channel]

    #         # calculate histogram for each colour and join to the result
    #         hist = np.concatenate(
    #             [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    hist = hist / len(c)

    return hist


def _calc_texture_gradient(img):
    """
        calculate texture gradient for entire image
        The original SelectiveSearch algorithm proposed Gaussian derivative
        for 8 orientations, but we use LBP instead.
        output will be [height(*)][width(*)]
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for colour_channel in [0]:
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    return ret


def _calc_texture_hist(img):
    """
        calculate texture histogram for each region
        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10

    hist = np.array([])

    c = img[:, :, 0].flatten()

    hist = np.histogram(c, BINS, (0.0, 255.0))[0]

    hist = hist / len(c)

    return hist


def _cal_confluency(binary_mask_patch):
    patch_w, patch_h = binary_mask_patch.shape
    cell_area_percent = np.sum(binary_mask_patch) / (patch_w * patch_h)

    return cell_area_percent


def _extract_regions(img, binary_mask, request_bbox_size, img_size):
    R = {}

    # get hsv image

    # hsv = skimage.color.rgb2hsv(img[:, :, :3])
    only_img = img[:, :, :3]
    # pass 1: count pixel positions

    for y, i in enumerate(img):

        for x, (r, g, b, l) in enumerate(i):

            if l != 0:
                # initialize a new region
                if (l not in R):
                    R[l] = {
                        "min_x": 0xffff, "min_y": 0xffff,
                        "max_x": 0, "max_y": 0, "labels": [l]}

                # bounding box
                if R[l]["min_x"] > x:
                    R[l]["min_x"] = x
                if R[l]["min_y"] > y:
                    R[l]["min_y"] = y
                if R[l]["max_x"] < x:
                    R[l]["max_x"] = x
                if R[l]["max_y"] < y:
                    R[l]["max_y"] = y

    current_region_num = R.keys()
    new_R = R.copy()

    region_w, region_h = request_bbox_size
    im_max_w, im_max_h = img_size

    for k, v in list(R.items()):
        x, y, w, h = R[k]['min_x'], R[k]['min_y'], R[k]['max_x'] - R[k]['min_x'], R[k]['max_y'] - R[k]['min_y']
        region_mask = binary_mask[R[k]['min_y']:R[k]['max_y'], R[k]['min_x']:R[k]['max_x']]
        if np.sum(region_mask) > request_bbox_size[0] * request_bbox_size[1] * 2:

            w_times, h_times = math.ceil(w / region_w), math.ceil(h / region_h)
            for i in range(0, h_times, 1):
                new_region_y = region_h * i + y
                for j in range(0, w_times, 1):
                    new_region_x = region_w * j + x

                    if new_region_x < 0:
                        new_region_x = 0
                    elif new_region_x + region_w > im_max_w:
                        new_region_x = im_max_w - region_w

                    if new_region_y < 0:
                        new_region_y = 0
                    elif new_region_y + region_h > im_max_h:
                        new_region_y = im_max_h - region_h

                    new_region_mask = binary_mask[new_region_y:new_region_y + region_h,
                                      new_region_x:new_region_x + region_w]
                    if _cal_confluency(new_region_mask) >= 0.0:
                        new_key = max(new_R.keys()) + 1

                        new_R[new_key] = {
                            "min_x": new_region_x, "min_y": new_region_y,
                            "max_x": new_region_x + region_w, "max_y": new_region_y + region_h, "labels": [new_key]}

    R = new_R
    # print('Initial region number is: ',len(R.keys()))
    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)

    # pass 3: calculate colour histogram of each region

    for k, v in list(R.items()):
        # region correction here
        x, y, w, h = R[k]['min_x'], R[k]['min_y'], R[k]['max_x'] - R[k]['min_x'], R[k]['max_y'] - R[k]['min_y']

        x, y, w, h = region_correction(x, y, w, h, mask=binary_mask, region_size=request_bbox_size, im_size=img_size)
        R[k]['min_x'], R[k]['min_y'], R[k]['max_x'], R[k]['max_y'] = x, y, x + w, y + h

        R[k]["confluency"] = _cal_confluency(binary_mask[R[k]['min_y']:R[k]['max_y'], R[k]['min_x']:R[k]['max_x']])

        # colour histogram
        # masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        zero_mask = np.zeros((img_size[1], img_size[0]))
        zero_mask[R[k]['min_y']:R[k]['max_y'], R[k]['min_x']:R[k]['max_x']] = 1

        masked_pixels = np.transpose(only_img, (2, 0, 1)) * zero_mask
        masked_pixels = masked_pixels.transpose((1, 2, 0))

        R[k]["size"] = R[k]["confluency"]
        R[k]["hist_c"] = _calc_colour_hist(only_img[R[k]['min_y']:R[k]['max_y'], R[k]['min_x']:R[k]['max_x'], :3])

        # texture histogram
        hist_t = np.transpose(tex_grad, (2, 0, 1)) * zero_mask
        hist_t = hist_t.transpose((1, 2, 0))

        R[k]["hist_t"] = _calc_texture_hist(tex_grad[R[k]['min_y']:R[k]['max_y'], R[k]['min_x']:R[k]['max_x']])

    return R


def _extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] <= b["min_x"] < a["max_x"] and a["min_y"] <= b["min_y"] < a["max_y"]) or (
                b["min_x"] <= a["min_x"] < b["max_x"] and b["min_y"] <= a["min_y"] < b["max_y"]) or (
                b["min_x"] <= a["min_x"] < b["max_x"] and a["min_y"] <= b["min_y"] < a["max_y"]) or (
                a["min_x"] <= b["min_x"] < a["max_x"] and b["min_y"] <= a["min_y"] < b["max_y"]):
            return True

        return False

    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def cluster_selection(
        im_orig, binary_mask, request_bbox_size, img_size, bb_num_limit=500, final_region=20):
    '''Selective Search
    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''
    # assert im_orig.shape[2] == 3, "3ch image is expected"

    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]

    binary_mask = binary_mask > 128

    img = _generate_segments(im_orig, binary_mask, request_bbox_size, bb_num_limit=bb_num_limit)

    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img, binary_mask, request_bbox_size, img_size)

    # extract neighbouring information
    neighbours = _extract_neighbours(R)

    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # overlap patch fliter
    # print(S)
    sim_values = [S[k] for k, v in S.items()]
    sim_mean = np.average(sim_values)

    key_to_delete = []
    for (i, j), sim in S.items():
        if (sim >= sim_mean) and (i not in key_to_delete) and (j not in key_to_delete):
            # if (i not in key_to_delete) and (j not in key_to_delete):
            if R[i]['confluency'] >= R[j]['confluency']:
                key_to_delete.append(j)
            else:
                key_to_delete.append(i)

    key_to_delete = np.unique(key_to_delete)
    for k in key_to_delete:
        del R[k]

    # assert len(R.keys()) >= final_region
    # confluency filter
    # print('region number is: ',len(R.keys()))
    R = sorted(R.items(), key=lambda i: i[1]['confluency'])[::-1]
    R = R[:final_region]

    regions = []
    for k, r in list(R):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'confluency': r['confluency'],
            'label': r['labels'],
        })

    return img, regions


def region_correction(x, y, w, h, mask, region_size, im_size):
    region_w, region_h = region_size
    im_max_w, im_max_h = im_size
    if (w == region_w) and (h == region_h):
        return x, y, w, h
    else:
        # for index
        w = w + 1
        h = h + 1

        if w == 0 and h != 0:
            new_x = x
            new_y = y + (h / 2)
        elif h == 0 and w != 0:
            new_x = x + (w / 2)
            new_y = y
        elif w == 0 and h == 0:
            new_x = x
            new_y = y
        else:
            # find center_of_mass
            new_mask = np.zeros((im_size[1], im_size[0]))
            new_mask[y:y + h, x:x + w] = 1

            new_mask = new_mask * mask

            cy, cx = ndi.center_of_mass(new_mask)

            new_x = cx - (112 / 2)
            new_y = cy - (112 / 2)

        new_x = int(new_x)
        new_y = int(new_y)

        '''for bbox locat near img boundary'''
        if new_x < 0:
            new_x = 0
        elif new_x + region_w > im_max_w:
            new_x = im_max_w - region_w

        if new_y < 0:
            new_y = 0
        elif new_y + region_h > im_max_h:
            new_y = im_max_h - region_h
        # print(cy,cx,w,h)
        return int(new_x), int(new_y), region_w, region_h




def find_patches(img_path, mask_path):
    im = skimage.io.imread(img_path, as_gray=True)
    # skimage shape 944*1280
    if im.shape == (944, 1280):
        im = resize(im, (1040, 1408), anti_aliasing=True)
    if len(im.shape) != 3:
        im = gray2rgb(im)
    im = im[:-150, :]

    request_bbox_size = (112, 112)
    bb_num_limit = 40
    final_region = 20
    img_size = (im.shape[1], im.shape[0])

    mask = skimage.io.imread(mask_path, as_gray=True)

    assert im.shape[0] == mask.shape[0] and im.shape[1] == mask.shape[1]

    img_lbl, regions = cluster_selection(
        im, mask, request_bbox_size, img_size, bb_num_limit=bb_num_limit, final_region=final_region)

    if len(regions) < 10:
        print('!!! {} region number: {}'.format(img_path, len(regions)))

    return regions


def load_batch_data(input_path,mask_path):

    cell_folder_names = natsorted(os.listdir(input_path))[:]


    print('we have {} cell lines'.format(len(cell_folder_names)))


    for folder_name in cell_folder_names:
        folder_path = os.path.join(input_path,folder_name)

        for batch_name in natsorted(os.listdir(folder_path))[:]:
            batch_path = os.path.join(folder_path,batch_name)
            print('Cell {} Batch {}'.format(folder_name,batch_name))
            mask_batch_path = os.path.join(mask_path,folder_name,batch_name)

            pl_dict = {}
            for img_name in tqdm(os.listdir(batch_path)):
                img_path = os.path.join(batch_path,img_name)
                img_mask_path = os.path.join(mask_batch_path,img_name)


                patches_location = find_patches(img_path,img_mask_path)

                pl_dict[img_name] = patches_location

            np.save(os.path.join(mask_batch_path,'patches.npy'),pl_dict)

    print('All masks have been generated')


if __name__ == '__main__':
    # Change the image_path to the download dataset path
    image_path = "/projects/img/cellbank/Cell_lines/"
    mask_path = "./image_masks/"

    load_batch_data(image_path, mask_path)