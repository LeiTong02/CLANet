import pathlib
import os
from skimage.transform import resize
from natsort import natsorted
import numpy as np

import skimage
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from skimage import feature, morphology, exposure
from skimage.filters import threshold_otsu
import cv2
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm


def remove_small_objs(mask, min_size=10 * 10):
    cv_image = img_as_ubyte(mask)
    contours, hierarchy = cv2.findContours(cv_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    des = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_size:
            cv2.drawContours(des, [cnt], 0, 255, -1)

    return des


def filling_holes(gray, num):


    result = np.array(gray, dtype=bool)
    result = ndi.binary_fill_holes(result, structure=np.ones((6, 6)))
    return result


def threshold_range(im):
    im_copy = im.copy()
    # im_copy = exposure.adjust_sigmoid(im_copy)

    threshold_sau = threshold_otsu(im_copy)

    im_copy = im_copy > threshold_sau
    return im_copy


def dilation_erosion(input_mask):
    dilation = binary_dilation(input_mask)
    erosion = binary_erosion(dilation)
    return erosion


def combine_two_masks(m1, m2):
    m1_np = m1
    m2_thresh_np = m2

    m1_np = dilation_erosion(m1_np)
    m2_thresh_np = dilation_erosion(m2_thresh_np)

    m_combined = np.logical_or(m1_np, m2_thresh_np)
    # return m_combined
    return filling_holes(m_combined.astype(int) * 255, num=4)


def combine_three_masks(m1, m2, m3):
    m1_np = dilation_erosion(m1)
    m2_np = dilation_erosion(m2)
    m3_np = dilation_erosion(m3)

    m_combined = np.logical_or(m1_np, m2_np)
    m_combined = np.logical_or(m_combined, m3_np)
    # return m_combined
    return filling_holes(m_combined.astype(int) * 255, num=4)


def threshold_canny_masks(im, im_enhanced, im_input_path):
    im_threshold_mask = threshold_range(im_enhanced)
    # Compute the Canny filter for two values of sigma
    mask_edges = feature.canny(im_enhanced)
    mask_diff_percent = np.mean(mask_edges != im_threshold_mask)
    diff_percent_thresh = 0.5
    if mask_diff_percent >= diff_percent_thresh:
        # for some images cell is lighter than background, bitwise the mask
        print('Bitwise the threshold mask!')
        im_threshold_mask = ~im_threshold_mask

        mask_diff_percent = np.mean(mask_edges != im_threshold_mask)
        if mask_diff_percent > 0.5:
            print('{} mask diff percent > 0.5'.format(im_input_path))


    return im_threshold_mask, mask_edges

def generate_masks_for_cell_images(img_path):
    im = skimage.io.imread(img_path, as_gray=True)

    # skimage shape 944*1280
    if im.shape == (944, 1280):
        im = resize(im, (1040, 1408), anti_aliasing=True)
    assert im.shape == (1040, 1408)
    im = im[:-150, :]

    p2, p98 = np.percentile(im, (0.2, 99.8))
    im_enhanced = exposure.rescale_intensity(im, in_range=(p2, p98))
    im_threshold_mask, mask_edges = threshold_canny_masks(im, im_enhanced, img_path)

    first_combined_mask = combine_two_masks(mask_edges, im_threshold_mask)
    combined_mask_array = [first_combined_mask]

    iteration_runs = 10
    tolerence_cost = 0.0001

    for i in range(iteration_runs):
        combined_mask = combine_three_masks(im_threshold_mask, mask_edges, combined_mask_array[i])
        cost_value = np.mean(combined_mask != combined_mask_array[i])
        combined_mask_array.append(combined_mask)
        # print('cost_value is {}'.format(cost_value))
        if cost_value < tolerence_cost:
            break

    clean_noise_mask = morphology.remove_small_objects(morphology.remove_small_holes(combined_mask_array[-1]), 10 * 10)
    # clean_noise_mask  = remove_small_objs(combined_mask_array[-1],min_size=10*10)
    return clean_noise_mask




def load_batch_data(input_path,output_path):

    cell_folder_names = os.listdir(input_path)

    print('we have {} cell lines'.format(len(cell_folder_names)))
    for folder_name in cell_folder_names:
        folder_path = os.path.join(input_path,folder_name)

        for batch_name in natsorted(os.listdir(folder_path))[1:]:
            batch_path = os.path.join(folder_path,batch_name)
            print('Cell {} Batch {}'.format(folder_name,batch_name))

            mask_batch_path = os.path.join(output_path,folder_name,batch_name)
            path_lib = pathlib.Path(mask_batch_path)
            path_lib.mkdir(parents=True, exist_ok=True)

            for img_name in tqdm(os.listdir(batch_path)):
                img_path = os.path.join(batch_path,img_name)
                clean_noise_mask = skimage.img_as_ubyte(generate_masks_for_cell_images(img_path))


                mask_path = os.path.join(mask_batch_path,img_name)


                cv2.imwrite(mask_path,clean_noise_mask)
    print('All masks have been generated')


if __name__ == '__main__':

    # Change the input_path to the download dataset path
    input_path = "/projects/img/cellbank/Cell_lines/"
    output = "./image_masks/"

    load_batch_data(input_path, output)