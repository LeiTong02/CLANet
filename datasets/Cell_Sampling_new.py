from __future__ import print_function, division


import numpy as np
import torch

from torch.utils.data import Dataset

import os,tqdm

from natsort import natsorted, ns
from scipy.special import softmax
# Ignore warnings
import scipy.stats as ss
import warnings
import random
import math
warnings.filterwarnings("ignore")

class Cell_Features_sampling(Dataset):

    def __init__(self, data_path,label_id_path_file=None,train=True,train_epoch=2000, transform=None,random_train_val=None):
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
        self.train_epoch=train_epoch
        self.class_number = 0
        if label_id_path_file==None:
            self.label_id_path_file = './train_class_idx.txt'
        else:
            self.label_id_path_file = label_id_path_file
        self.transform = transform


        '''Parameter for Segment Sampling'''
        '''t_gap: segment length [0,t_gap)'''
        self.accum_value = 0

        #norm  distribution pick up t_gap
        self.t_gap = np.arange(1,9,1)
        self.img_percent_coef = 1
        self.sparsity_coef = 1
        self.random_train_val=random_train_val
        self.load_data()

    def get_id_file(self):
        return self.label_id_path_file


    def load_data(self):

        with open(self.label_id_path_file,'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        with open(self.random_train_val,'r') as f:
            b_lines = f.readlines()
            b_lines = [line.rstrip() for line in b_lines]

        self.sample_vectors, self.labels_idx = [],[]
        self.batch_token, batch_token_count = [], 0
        self.batch_segment_times = []
        self.norm_t_gaps,self.expect_t = [],[]
        for label_id, class_name in tqdm.tqdm(enumerate(lines),total = len(lines)):
            batch_number = len(os.listdir(os.path.join(self.data_path, class_name)))
            assert batch_number!=0

            train_b_id = int(b_lines[label_id])
            if self.train:
                b_range = [train_b_id]
            else:
                b_range = list(range(1, batch_number + 1, 1))
                b_range.remove(train_b_id)

            for i in b_range:
                npy_path = os.path.join(self.data_path, class_name, 'batch_{}'.format(i),'patch_features.npy')
                patch_dict = np.load(npy_path, allow_pickle=True).item()

                sorted_img_names = natsorted(list(patch_dict.keys()))

                img_positions = []
                position_id_end = [0]
                p_token = 1
                batch_features = []
                bag_times = []
                position_count = 0
                for img_idx, img_name in enumerate(sorted_img_names):
                    filename = img_name.split('.')[0]
                    flask_position_name = filename.split('_')[1]
                    if class_name == 'HS-5' and i == 2:
                        flask_position_name = filename.split('_')[2]
                    # '00d12h00m'
                    file_cTime = filename[-9:]
                    # To hour-style ~
                    label_time = (int(file_cTime[0:2]) * 24) + int(file_cTime[3:5])+math.ceil(float(file_cTime[6:8]) / 60)

                    if len(patch_dict[img_name]) < 10:
                        continue
                    if isinstance(patch_dict[img_name],np.ndarray):
                        features = [patch_dict[img_name]]
                    else:

                        patches_list = sorted(patch_dict[img_name], key=lambda i: i['confluency'])[::-1]


                        features = [r['prediction'] for r in patches_list[:10]]


                    batch_features.append(features)
                    bag_times.append(label_time)
                    img_positions.append(flask_position_name)
                    position_count += 1
                    if len(np.unique(img_positions)) > p_token:
                        p_token = len(np.unique(img_positions))
                        position_id_end.append(position_count - 1)

                position_id_end.append(position_count)

                assert len(position_id_end) -1 == len(np.unique(img_positions))

                bag_features = []
                bag_segment_times = []
                norm_sampling_t_gaps = []
                e_t_list = []


                for seq_idx, p_id in enumerate(range(len(position_id_end))[1:]):
                    # seq_num x 10 x 2048
                    if len(bag_times[position_id_end[p_id-1]:position_id_end[p_id]]) < 3:
                        print(npy_path)
                        continue

                    #bag_features.append(np.reshape(batch_features[position_id_end[p_id-1]:position_id_end[p_id]], (-1,1536)))
                    bag_features.append(np.array(batch_features[position_id_end[p_id - 1]:position_id_end[p_id]]))
                    s_times = bag_times[position_id_end[p_id-1]:position_id_end[p_id]]
                    bag_segment_times.append(s_times)


                    if self.train:
                        sum_t_gap = 0
                        for t_id in range(len(s_times))[1:]:
                            # if s_times[t_id] <= s_times[t_id - 1]:
                            #     print(npy_path)
                            sum_t_gap += (s_times[t_id] - s_times[t_id - 1])
                        expectation_t = sum_t_gap / (len(s_times) - 1)
                        if expectation_t<self.t_gap[0]:
                            expectation_t=self.t_gap[0]
                        elif expectation_t>self.t_gap[-1]:
                            expectation_t = self.t_gap[-1]

                        prob = ss.norm.pdf(self.t_gap,loc=expectation_t,scale=np.std(self.t_gap))
                        prob = prob/prob.sum()
                        nums = np.random.choice(self.t_gap,size=self.train_epoch,p=prob)

                        '''The first 5 epoch will be the original feature list for model training in order to keep training stable'''
                        e_idx = np.where(nums==expectation_t)[0][:5]
                        nums[e_idx], nums[:len(e_idx)] = nums[:len(e_idx)], nums[e_idx]

                        norm_sampling_t_gaps.append(nums)
                        e_t_list.append(expectation_t)





                self.sample_vectors.extend(bag_features)
                self.labels_idx.extend([label_id]*len(bag_features))
                self.batch_token.extend([batch_token_count]*len(bag_features))
                self.batch_segment_times.extend(bag_segment_times)
                self.norm_t_gaps.extend(norm_sampling_t_gaps)
                self.expect_t.extend(e_t_list)
                batch_token_count+=1

        self.class_number = len(lines)
        print('Dataset includes {} classes with {} flasks {} samples.'.format(self.class_number,batch_token_count,len(self.sample_vectors)))


    def __len__(self):
        return len(self.labels_idx)

    def get_sample_weights(self,original_img_num,sampled_img_num,sampled_times,expectation_t,t_gaps):
        sum_t_gap = 0

        if expectation_t<t_gaps:
            for t in range(len(sampled_times))[1:]:
                t_g = sampled_times[t] - sampled_times[t-1]
                'original give a specific value 12 here'
                if t_g > t_gaps:
                    sum_t_gap+=0
                else:
                    sum_t_gap=sum_t_gap+((t_g/expectation_t)-1)

            img_percent_weight = (sampled_img_num/original_img_num)
            sparsity_weight = (sum_t_gap/(((t_gaps/expectation_t)-1)*(sampled_img_num-1)))

            if sparsity_weight>1:
                sparsity_weight =1
            elif sparsity_weight<0:
                sparsity_weight = 0


            weights = self.img_percent_coef*img_percent_weight + self.sparsity_coef*sparsity_weight
        else:

            img_percent_weight = (original_img_num/sampled_img_num)
            weights = self.img_percent_coef*img_percent_weight

        return weights
    def __getitem__(self, index):

        img, target = self.sample_vectors[index], self.labels_idx[index]
        times = np.array(self.batch_segment_times[index])

        if self.train:
            #Use 12h as a segment
            position_id_end = [0]
            #           norm_t_gap 2000d norm nums

            t_gaps, expectation_t = self.norm_t_gaps[index][int(self.accum_value/len(self.labels_idx))],self.expect_t[index]
            self.accum_value+=1

            t_gap_count = t_gaps

            assert len(img) == len(times)

            if t_gaps <= round(expectation_t):
                new_imgs =torch.from_numpy(np.array(img))
                sample_weights=1

            else:
                for t_id in range(len(times))[1:]:
                    if int(times[t_id]) > (t_gap_count):
                        position_id_end.append(t_id)
                        while (t_gap_count) < int(times[t_id]):
                            t_gap_count += t_gaps
                    elif int(times[t_id]) == (t_gap_count):
                        position_id_end.append(t_id + 1)
                        if (t_id + 1) < len(img):
                            t_gap_count = int(times[t_id + 1]) + t_gaps

                if position_id_end[-1] < len(img):
                    position_id_end.append(len(img))

                sampling_segment = []
                sampling_times = []

                while len(sampling_segment)<1:

                    sampling_segment = []
                    sampling_times = []
                    for p_id in range(len(position_id_end))[1:]:
                        imgs_seg = img[position_id_end[p_id-1]:position_id_end[p_id]]
                        time_seg = times[position_id_end[p_id-1]:position_id_end[p_id]]


                        if len(imgs_seg)>1:
                            rand_img_num = np.random.randint(low=1, high=len(imgs_seg)+1, size=1)

                        else:
                            #(0,1)
                            rand_img_num = np.random.randint(low=0, high=2, size=1)

                        rand_img_idx = np.sort(np.random.choice(len(imgs_seg),size=rand_img_num,replace=False))


                        sampling_segment.extend(imgs_seg[rand_img_idx])
                        sampling_times.extend(time_seg[rand_img_idx])


                new_imgs = torch.from_numpy(np.array(sampling_segment))

                sample_weights=self.get_sample_weights(len(img),len(sampling_segment),sampling_times,round(expectation_t),t_gaps)
                #print('sample_weights:',sample_weights)


        else:

            new_imgs = torch.from_numpy(np.array(img))
            sample_weights=1

        return new_imgs,torch.tensor([target]), torch.tensor([sample_weights])


import pandas as pd
import ast

class Cell_Features_sampling_batch_stratified(Dataset):

    def __init__(self, data_path,label_id_path_file=None,train=True,train_epoch=2000, transform=None,random_train_val=None):
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
        self.train_epoch = train_epoch
        self.class_number = 0
        if label_id_path_file==None:
            self.label_id_path_file = './train_class_idx.txt'
        else:
            self.label_id_path_file = label_id_path_file
        self.transform = transform


        '''Parameter for Segment Sampling'''
        '''t_gap: segment length [0,t_gap)'''
        self.accum_value = 0


        #norm  distribution pick up t_gap
        self.t_gap = np.arange(1,9,1)
        self.img_percent_coef = 1
        self.sparsity_coef = 1
        self.random_train_val=random_train_val
        self.load_data()

    def get_id_file(self):
        return self.label_id_path_file


    def load_data(self):

        with open(self.label_id_path_file,'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        '''Here, random_train_val is a id'''


        if self.train:
            csv_path = '/projects/img/cellbank/Cell_Batch_Classification/experiment_setup/batch_stratified/random{}.csv'.format(
                self.random_train_val)
            df = pd.read_csv(csv_path, index_col=0)
        else:
            csv_path = '/projects/img/cellbank/Cell_Batch_Classification/experiment_setup/batch_stratified/random{}_test.csv'.format(
                self.random_train_val)
            df = pd.read_csv(csv_path, index_col=0)

        self.sample_vectors, self.labels_idx = [],[]
        self.batch_token, batch_token_count = [], 0
        self.batch_segment_times = []
        self.norm_t_gaps,self.expect_t = [],[]
        for label_id, class_name in tqdm.tqdm(enumerate(lines),total = len(lines)):

            batch_number = len(os.listdir(os.path.join(self.data_path, class_name)))
            assert batch_number!=0
            '''Like array(["['B3']", "['C6', 'E4', 'E10', 'G2']"], dtype=object)'''
            posi_array=[]
            for string_position in df.loc[label_id].dropna().values:
                posi_array.append(ast.literal_eval(string_position))



            for i,split_posi in enumerate(posi_array):
                if len(split_posi) is not 0:

                    npy_path = os.path.join(self.data_path, class_name, 'batch_{}'.format(i+1),'patch_features.npy')
                    patch_dict = np.load(npy_path, allow_pickle=True).item()

                    sorted_img_names = natsorted([elem for elem in list(patch_dict.keys()) if any(x in elem for x in split_posi)])

                    img_positions = []
                    position_id_end = [0]
                    p_token = 1
                    batch_features = []
                    bag_times = []
                    position_count = 0
                    for img_idx, img_name in enumerate(sorted_img_names):
                        filename = img_name.split('.')[0]
                        flask_position_name = filename.split('_')[1]
                        # '00d12h00m'
                        file_cTime = filename[-9:]
                        # To hour-style ~
                        label_time = (int(file_cTime[0:2]) * 24) + int(file_cTime[3:5])+math.ceil(float(file_cTime[6:8]) / 60)

                        if len(patch_dict[img_name]) < 10:
                            continue


                        patches_list = sorted(patch_dict[img_name], key=lambda i: i['confluency'])[::-1]


                        features = [r['prediction'] for r in patches_list[:10]]

                        batch_features.append(features)
                        bag_times.append(label_time)
                        img_positions.append(flask_position_name)
                        position_count += 1
                        if len(np.unique(img_positions)) > p_token:
                            p_token = len(np.unique(img_positions))
                            position_id_end.append(position_count - 1)

                    position_id_end.append(position_count)

                    assert len(position_id_end) -1 == len(np.unique(img_positions))

                    bag_features = []
                    bag_segment_times = []
                    norm_sampling_t_gaps = []
                    e_t_list = []


                    for seq_idx, p_id in enumerate(range(len(position_id_end))[1:]):
                        # seq_num x 10 x 2048
                        if len(bag_times[position_id_end[p_id-1]:position_id_end[p_id]]) < 3:
                            print(npy_path)
                            continue

                        #bag_features.append(np.reshape(batch_features[position_id_end[p_id-1]:position_id_end[p_id]], (-1,1536)))
                        bag_features.append(np.array(batch_features[position_id_end[p_id - 1]:position_id_end[p_id]]))
                        s_times = bag_times[position_id_end[p_id-1]:position_id_end[p_id]]
                        bag_segment_times.append(s_times)


                        if self.train:
                            sum_t_gap = 0
                            for t_id in range(len(s_times))[1:]:
                                # if s_times[t_id] <= s_times[t_id - 1]:
                                #     print(npy_path)
                                sum_t_gap += (s_times[t_id] - s_times[t_id - 1])
                            expectation_t = sum_t_gap / (len(s_times) - 1)
                            if expectation_t<self.t_gap[0]:
                                expectation_t=self.t_gap[0]
                            elif expectation_t>self.t_gap[-1]:
                                expectation_t = self.t_gap[-1]

                            prob = ss.norm.pdf(self.t_gap,loc=expectation_t,scale=np.std(self.t_gap))
                            prob = prob/prob.sum()
                            nums = np.random.choice(self.t_gap,size=self.train_epoch,p=prob)

                            '''The first 5 epoch will be the original feature list for model training in order to keep training stable'''
                            e_idx = np.where(nums==expectation_t)[0][:5]
                            nums[e_idx], nums[:len(e_idx)] = nums[:len(e_idx)], nums[e_idx]

                            norm_sampling_t_gaps.append(nums)
                            e_t_list.append(expectation_t)





                    self.sample_vectors.extend(bag_features)
                    self.labels_idx.extend([label_id]*len(bag_features))
                    self.batch_token.extend([batch_token_count]*len(bag_features))
                    self.batch_segment_times.extend(bag_segment_times)
                    self.norm_t_gaps.extend(norm_sampling_t_gaps)
                    self.expect_t.extend(e_t_list)
                    batch_token_count+=1

        self.class_number = len(lines)
        print('Dataset includes {} classes with {} flasks {} samples.'.format(self.class_number,batch_token_count,len(self.sample_vectors)))


    def __len__(self):
        return len(self.labels_idx)

    def get_sample_weights(self,original_img_num,sampled_img_num,sampled_times,expectation_t,t_gaps):
        sum_t_gap = 0

        if expectation_t<t_gaps:
            for t in range(len(sampled_times))[1:]:
                t_g = sampled_times[t] - sampled_times[t-1]
                'original give a specific value 12 here'
                if t_g > t_gaps:
                    sum_t_gap+=0
                else:
                    sum_t_gap=sum_t_gap+((t_g/expectation_t)-1)

            img_percent_weight = (sampled_img_num/original_img_num)
            sparsity_weight = (sum_t_gap/(((t_gaps/expectation_t)-1)*(sampled_img_num-1)))

            if sparsity_weight>1:
                sparsity_weight =1
            elif sparsity_weight<0:
                sparsity_weight = 0


            weights = self.img_percent_coef*img_percent_weight + self.sparsity_coef*sparsity_weight
        else:

            img_percent_weight = (original_img_num/sampled_img_num)
            weights = self.img_percent_coef*img_percent_weight

        return weights
    def __getitem__(self, index):

        img, target = self.sample_vectors[index], self.labels_idx[index]
        times = np.array(self.batch_segment_times[index])
        new_imgs = []
        sample_weights = []
        if self.train:
            #Use 12h as a segment
            position_id_end = [0]
            #           norm_t_gap 2000d norm nums

            t_gaps, expectation_t = self.norm_t_gaps[index][int(self.accum_value/len(self.labels_idx))],self.expect_t[index]
            self.accum_value+=1

            t_gap_count = t_gaps

            assert len(img) == len(times)

            if t_gaps <= round(expectation_t):
                new_imgs =torch.from_numpy(np.array(img))
                sample_weights=1

            else:
                for t_id in range(len(times))[1:]:
                    if int(times[t_id]) > (t_gap_count):
                        position_id_end.append(t_id)
                        while (t_gap_count) < int(times[t_id]):
                            t_gap_count += t_gaps
                    elif int(times[t_id]) == (t_gap_count):
                        position_id_end.append(t_id + 1)
                        if (t_id + 1) < len(img):
                            t_gap_count = int(times[t_id + 1]) + t_gaps

                if position_id_end[-1] < len(img):
                    position_id_end.append(len(img))

                sampling_segment = []
                sampling_times = []

                while len(sampling_segment)<1:

                    sampling_segment = []
                    sampling_times = []
                    for p_id in range(len(position_id_end))[1:]:
                        imgs_seg = img[position_id_end[p_id-1]:position_id_end[p_id]]
                        time_seg = times[position_id_end[p_id-1]:position_id_end[p_id]]


                        if len(imgs_seg)>1:
                            rand_img_num = np.random.randint(low=1, high=len(imgs_seg)+1, size=1)
                            #rand_img_idx = np.sort(np.random.choice(len(imgs_seg), size=rand_img_num, replace=False))
                        else:
                            #(0,1)
                            rand_img_num = np.random.randint(low=0, high=2, size=1)
                            # rand_img_idx = np.sort(np.random.choice(len(imgs_seg), size=rand_img_num, replace=True))
                        rand_img_idx = np.sort(np.random.choice(len(imgs_seg),size=rand_img_num,replace=False))


                        sampling_segment.extend(imgs_seg[rand_img_idx])
                        sampling_times.extend(time_seg[rand_img_idx])


                new_imgs = torch.from_numpy(np.array(sampling_segment))

                sample_weights=self.get_sample_weights(len(img),len(sampling_segment),sampling_times,round(expectation_t),t_gaps)
                #print('sample_weights:',sample_weights)


        else:
            new_imgs = torch.from_numpy(np.array(img))
            sample_weights=1

        return new_imgs,torch.tensor([target]), torch.tensor([sample_weights])




