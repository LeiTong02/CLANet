from __future__ import print_function, division
from PIL import Image
from scipy.ndimage import label
import numpy as np
import torch
from skimage.segmentation import felzenszwalb
from skimage.color import gray2rgb
from torch.utils.data import Dataset

import skimage
import os, tqdm
import random
from natsort import natsorted, ns
from torchvision import transforms
# Ignore warnings
from skimage.transform import resize
from einops import rearrange, reduce, repeat
import warnings

warnings.filterwarnings("ignore")

class Cell_Images(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, label_id_path_file=None, train=True,sup_train=False, transform=None, shuffle=False, remove_times=0
                 , selected_cell_data=None,batch_id=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            remove_times(int): remove images which time point < parameter
        """
        self.data_path = data_path

        self.train = train
        self.transform = transform
        self.sup_train = sup_train

        self.class_number = 0
        if label_id_path_file == None:
            self.label_id_path_file = './train_class_idx.txt'
        else:
            self.label_id_path_file = label_id_path_file
        self.shuffle = shuffle
        self.remove_times = remove_times
        self.selected_cell_data = selected_cell_data
        self.patch_npy_root =  '/projects/img/cellbank/Cell_Batch_Classification/enhanced_data_mask/'
        self.batch_id = batch_id
        self.load_data()

    def get_id_file(self):
        return self.label_id_path_file

    def get_file(self, img_path,patch_path, img_sequence_num = 3):

        img_names = os.listdir(img_path)
        sorted_img_names = natsorted(img_names)
        img_list = []
        patch_list = []
        img_positions = []

        for img_name in sorted_img_names[:]:

            filename = img_name.split('.')[0]
            flask_position_name = filename.split('_')[1]
            img_positions.append(flask_position_name)
            if len(np.unique(img_positions)) <= img_sequence_num and self.train:
                img_list.append(os.path.join(img_path,img_name))

            elif self.train is not True:
                img_list.append(os.path.join(img_path, img_name))

            else:
                break
        if len(img_list) < 100:
            print(img_path)
            #pass
        return img_list, patch_list

    def load_data(self):
        categories = []
        lines = []
        patches = []
        with open(self.label_id_path_file, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # select one cell lines
        if self.selected_cell_data is not None and self.selected_cell_data in lines:


            class_dir_path = os.path.join(self.data_path, self.selected_cell_data, 'batch_{}'.format(self.batch_id))
            assert os.path.isdir(class_dir_path)
            img_paths, patch_list = self.get_file(img_path=class_dir_path
                                                  , patch_path=os.path.join(self.patch_npy_root, self.selected_cell_data,
                                                                            'batch_{}'.format(self.batch_id), "patches.npy"))
            categories.extend(img_paths)
            patches.extend(patch_list)

        else:
            for class_name in lines:
                batch_number = len(os.listdir(os.path.join(self.data_path, class_name)))

                if self.sup_train == True :
                    b_range = range(1, 2, 1)
                elif self.sup_train == False and self.train==False:
                    b_range = range(2, batch_number + 1, 1)
                elif self.sup_train == False and self.train==True:
                    b_range = range(1, batch_number + 1, 1)

                for i in b_range:
                    class_dir_path = os.path.join(self.data_path, class_name, 'batch_{}'.format(i))

                    if os.path.isdir(class_dir_path):
                        img_paths,patch_list= self.get_file(img_path=class_dir_path
                                                  , patch_path=os.path.join(self.patch_npy_root,class_name,'batch_{}'.format(i),"patches.npy"))
                        categories.extend(img_paths)
                        patches.extend(patch_list)
                    else:
                        print(class_dir_path)
        data_list = categories


        self.images_data, self.labels_idx, self.labels = [], [], []
        self.label_times = []
        self.unique_image_tokens = []
        self.img_masks = []
        self.confluency = []

        with_platform = os.name

        for file_index in tqdm.tqdm(range(len(data_list))):
            file_path = data_list[file_index]
            img = skimage.io.imread(file_path,as_gray=True)

            if img.shape == (944, 1280):
                img = resize(img, (1040, 1408), anti_aliasing=True)

            #Center_Crop

            img = img[:890,259:1149]


            if len(img.shape) != 3:
                img= gray2rgb(img)
            img = img*255
            img = img.astype(np.uint8)

            img = Image.fromarray(img)

            candidates = []
            candidates.append(img)


            # /cell_line/batch_1/image_names
            if with_platform == 'posix':
                label = file_path.split('/')[-3]
            elif with_platform == 'nt':
                label = file_path.split('\\')[-3]
            if with_platform == 'posix':
                filename = file_path.split('/')[-1]
            elif with_platform == 'nt':
                filename = file_path.split('\\')[-1]

            self.images_data.extend(candidates)

            self.labels.extend([label for i in np.arange(len(candidates))])
            self.unique_image_tokens.extend([file_index for i in np.arange(len(candidates))])



        for label in self.labels:

            assert label.rstrip() in lines, "{} is not trained cell lines";
            if label.rstrip() in lines:
                idx = lines.index(label.rstrip())
                self.labels_idx.append(idx)

        self.class_number = np.unique(self.labels_idx).shape[0]
        print('Cell Batch includes {} classes with totally {} images.'.format(self.class_number,
                                                                                 len(self.images_data)))


    def __len__(self):
        return len(self.labels_idx)

    def __getitem__(self, index):

        img, target = self.images_data[index], self.labels_idx[index]

        # scale the range of time label


        return self.transform(img), target






class Cell_masked_patches(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, label_id_path_file=None, train=True,sup_train=False, transform=None, shuffle=False, remove_times=0
                 , selected_cell_data=None,batch_id=None,patch_npy_root=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            remove_times(int): remove images which time point < parameter
        """
        self.data_path = data_path

        self.train = train
        self.transform = transform
        self.sup_train = sup_train

        self.class_number = 0
        if label_id_path_file == None:
            self.label_id_path_file = './train_class_idx.txt'
        else:
            self.label_id_path_file = label_id_path_file
        self.shuffle = shuffle
        self.remove_times = remove_times
        self.selected_cell_data = selected_cell_data
        self.patch_npy_root =  patch_npy_root
        self.batch_id = batch_id
        self.load_data()

    def get_id_file(self):
        return self.label_id_path_file

    def get_file(self, img_path,patch_path, img_sequence_num = 7):

        # img_names = os.listdir(img_path)
        # sorted_img_names = natsorted(img_names)
        img_list = []
        patch_list = []
        img_positions = []
        patch_dict = np.load(patch_path,allow_pickle=True).item()

        img_names = patch_dict.keys()
        sorted_img_names = natsorted(img_names)
        for img_name in sorted_img_names[:]:

            filename = img_name.split('.')[0]
            flask_position_name = filename.split('_')[1]
            img_positions.append(flask_position_name)
            if len(np.unique(img_positions)) <= img_sequence_num and self.train:
                img_list.append(os.path.join(img_path,img_name))
                patch_list.append(patch_dict[img_name])
            elif self.train is not True:
                img_list.append(os.path.join(img_path, img_name))
                patch_list.append(patch_dict[img_name])
            else:
                break
        if len(img_list) < 100:
            print(img_path)

        return img_list, patch_list

    def load_data(self):
        categories = []
        lines = []
        patches = []
        with open(self.label_id_path_file, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # select one cell lines
        if self.selected_cell_data is not None and self.selected_cell_data in lines:


            class_dir_path = os.path.join(self.data_path, self.selected_cell_data, 'batch_{}'.format(self.batch_id))
            assert os.path.isdir(class_dir_path)
            img_paths, patch_list = self.get_file(img_path=class_dir_path
                                                  , patch_path=os.path.join(self.patch_npy_root, self.selected_cell_data,
                                                                            'batch_{}'.format(self.batch_id), "patches.npy"))
            categories.extend(img_paths)
            patches.extend(patch_list)

        else:
            for class_name in lines:
                batch_number = len(os.listdir(os.path.join(self.data_path, class_name)))

                if self.sup_train == True :
                    b_range = range(1, 2, 1)
                elif self.sup_train == False and self.train==False:
                    b_range = range(2, batch_number + 1, 1)
                elif self.sup_train == False and self.train==True:
                    b_range = range(1, batch_number + 1, 1)

                for i in b_range:
                    class_dir_path = os.path.join(self.data_path, class_name, 'batch_{}'.format(i))

                    if os.path.isdir(class_dir_path):
                        img_paths,patch_list= self.get_file(img_path=class_dir_path
                                                  , patch_path=os.path.join(self.patch_npy_root,class_name,'batch_{}'.format(i),"patches.npy"))
                        categories.extend(img_paths)
                        patches.extend(patch_list)
                    else:
                        print(class_dir_path)
        data_list = categories


        self.images_data, self.labels_idx, self.labels = [], [], []
        self.label_times = []
        self.unique_image_tokens = []
        self.img_masks = []
        self.confluency = []

        with_platform = os.name

        for file_index in tqdm.tqdm(range(len(data_list))):
            file_path = data_list[file_index]
            img = skimage.io.imread(file_path,as_gray=True)

            if img.shape == (944, 1280):
                img = resize(img, (1040, 1408), anti_aliasing=True)

            if len(img.shape) != 3:
                img= gray2rgb(img)
            img = img*255
            img = img.astype(np.uint8)

            img_patch_regions = patches[file_index]

            candidates = []
            if self.train:
                if len(img_patch_regions) in [0,1]:
                    continue

                for r in img_patch_regions[:10]:
                    # excluding same rectangle (with different segments)
                    x, y, w, h = r['rect']

                    candidates.append(Image.fromarray(img[y:y+h,x:x+w,:]))
            else:
                if len(img_patch_regions) in [0,1]:
                    continue



                    return x,y,new_w,new_w
                for r in img_patch_regions[:]:
                    # excluding same rectangle (with different segments)
                    x, y, w, h = r['rect']

                    candidates.append(Image.fromarray(img[y:y + h, x:x + w, :]))


            # /cell_line/batch_1/image_names
            if with_platform == 'posix':
                label = file_path.split('/')[-3]
            elif with_platform == 'nt':
                label = file_path.split('\\')[-3]
            if with_platform == 'posix':
                filename = file_path.split('/')[-1]
            elif with_platform == 'nt':
                filename = file_path.split('\\')[-1]

            self.images_data.extend(candidates)

            self.labels.extend([label for i in np.arange(len(candidates))])
            self.unique_image_tokens.extend([file_index for i in np.arange(len(candidates))])



        for label in self.labels:

            assert label.rstrip() in lines, "{} is not trained cell lines";
            if label.rstrip() in lines:
                idx = lines.index(label.rstrip())
                self.labels_idx.append(idx)

        self.class_number = np.unique(self.labels_idx).shape[0]
        print('Cell Batch includes {} classes with totally {} images.'.format(self.class_number,
                                                                                 len(self.images_data)))

    def __len__(self):
        return len(self.labels_idx)

    def __getitem__(self, index):

        img, target = self.images_data[index], self.labels_idx[index]

        # scale the range of time label
        return self.transform(img), target








