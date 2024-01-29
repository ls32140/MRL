# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

import cv2
from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import json
from utils.config import args
from numpy.testing import assert_array_almost_equal
import h5py
logger = getLogger()

class cross_modal_dataset(data.Dataset):
    def __init__(self, dataset, noisy_ratio, mode, noise_mode='sym', root_dir='data/', noise_file=None, pred=False, probability=[], log=''):
        self.r = noisy_ratio # noise ratio
        self.mode = mode
        if 'wiki' in dataset.lower():
            root_dir = os.path.join(root_dir, 'wiki')
            path = os.path.join(root_dir, 'wiki_clip.mat')
            # path = os.path.join(root_dir, 'wiki_deep_doc2vec_data_corr_ae.h5py')  # wiki_deep_doc2vec_data
        elif 'nus' in dataset.lower():
            root_dir = os.path.join(root_dir, 'nus')
            path = os.path.join(root_dir, 'nus_clip.mat')
            valid_len = 5000
        elif 'inria' in dataset.lower():
            root_dir = os.path.join(root_dir, 'INRIA-Websearch')
            path = os.path.join(root_dir, 'inria6.mat')
            valid_len = 1332
        elif 'xmedianet' in dataset.lower():
            root_dir = os.path.join(root_dir, 'xmedianet')
            path = os.path.join(root_dir, 'xmedianet_clip.mat')
            valid_len = 4000
        elif 'ps' in dataset.lower():
            root_dir = os.path.join(root_dir, 'ps')
            path = os.path.join(root_dir, 'ps_clip200.mat')
        else:
            raise Exception('Have no such dataset!')
        data = sio.loadmat(path)
        if 'inria' in dataset.lower():
            if self.mode == 'train':
                train_data = [data['tr_img'][valid_len:].astype('float32'),
                              data['tr_text'][valid_len:].astype('float32')]
                train_label = [data['tr_label'][valid_len:].reshape([-1]).astype('int64'),
                               data['tr_label'][valid_len:].reshape([-1]).astype('int64')]
            elif self.mode == 'valid':
                train_data = [data['tr_img'][0: valid_len].astype('float32'),
                              data['tr_text'][0: valid_len].astype('float32')]
                train_label = [data['tr_label'][0: valid_len].reshape([-1]).astype('int64'),
                               data['tr_label'][0: valid_len].reshape([-1]).astype('int64')]
            elif self.mode == 'test':
                train_data = [data['te_img'].astype('float32'), data['te_text'].astype('float32')]
                train_label = [data['te_label'].reshape([-1]).astype('int64'),
                               data['te_label'].reshape([-1]).astype('int64')]
            else:
                raise Exception('Have no such set mode!')
        else:
            if self.mode == 'train':
                train_data = [data['tr_fc7'].astype('float32'),
                              data['tr_text'].astype('float32')]
                train_label = [data['tr_label'].reshape([-1]).astype('int64'),
                               data['tr_label'].reshape([-1]).astype('int64')]
            elif self.mode == 'valid':
                train_data = [data['te_fc7'].astype('float32'),
                              data['te_text'].astype('float32')]
                train_label = [data['te_label'].reshape([-1]).astype('int64'),
                               data['te_label'].reshape([-1]).astype('int64')]
            elif self.mode == 'test':
                train_data = [data['te_fc7'].astype('float32'),
                              data['te_text'].astype('float32')]
                train_label = [data['te_label'].reshape([-1]).astype('int64'),
                               data['te_label'].reshape([-1]).astype('int64')]

        self.train_label = [la.astype('int64') for la in train_label]
        noise_label = self.train_label
        if noise_file is None:
            if noise_mode == 'sym':
                noise_file = os.path.join(root_dir, 'noise_labels_%g_sym.json' % self.r)
            elif noise_mode == 'asym':
                noise_file = os.path.join(root_dir, 'noise_labels_%g__asym.json' % self.r)
        if self.mode == 'train':
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
                self.class_num = np.unique(noise_label).shape[0]
            else:    #inject noise
                noise_label = []
                classes = np.unique(train_label[0])
                class_num = classes.shape[0]
                self.class_num = class_num
                inx = np.arange(class_num)
                np.random.shuffle(inx)
                self.transition = {i: i for i in range(class_num)}
                half_num = int(class_num // 2)
                for i in range(half_num):
                    self.transition[inx[i]] = int(inx[half_num + i])
                for v in range(len(train_data)):
                    noise_label_tmp = []
                    data_num = train_data[v].shape[0]
                    idx = list(range(data_num))
                    random.shuffle(idx)
                    num_noise = int(self.r * data_num)
                    noise_idx = idx[:num_noise]
                    for i in range(data_num):
                        if i in noise_idx:
                            if noise_mode == 'sym':
                                noiselabel = int(random.randint(0, class_num-1))
                                noise_label_tmp.append(noiselabel)
                            elif noise_mode == 'asym':
                                noiselabel = self.transition[train_label[v][i]]
                                noise_label_tmp.append(noiselabel)
                        else:
                            noise_label_tmp.append(int(train_label[v][i]))
                    noise_label.append(noise_label_tmp)
                # print("save noisy labels to %s ..." % noise_file)
                json.dump(noise_label, open(noise_file, "w"))

        self.default_train_data = train_data
        self.default_noise_label = np.array(noise_label)
        self.train_data = self.default_train_data
        self.noise_label = self.default_noise_label
        if pred:
            self.prob = [np.ones_like(ll) for ll in self.default_noise_label]
        else:
            self.prob = None

    # 选中的数据，有多少是干净的 （选中干净/选中）
    def testClean(self, sidx, cidx, correct_result, hidx):
        n_view = len(self.train_data)
        select = []
        correct = []
        hard = []
        for v in range(n_view):
            sid = sidx[v].cpu()
            a = self.noise_label[v][sid] - self.train_label[v][sid]
            select.append(a)

        # select_hstack = np.hstack(select)
        # select_array = np.where(select_hstack, 0, 1)
        # num = np.sum(select_array)
        # rio = num / len(select_hstack)
        # print(f"选中干净/选中: {rio:.4f} ({num} / {len(select_hstack)})")
        select_hstack = np.hstack(select)
        select_array1 = np.where(select[0], 0, 1)
        select_array2 = np.where(select[1], 0, 1)
        num1 = np.sum(select_array1)
        num2 = np.sum(select_array2)
        num = num1+num2
        rio = num / len(select_hstack)
        print(f"选中干净/选中: {rio:.4f} ({num1 } / {num2})")

        for v in range(n_view):
            cid = cidx[v].cpu()
            result = correct_result[v].cpu()
            # print(result)
            b = result - self.train_label[v][cid]
            correct.append(b)
        correct_hstack = np.hstack(correct)
        correct_array = np.where(correct_hstack, 0, 1)
        num = np.sum(correct_array)
        rio = num / len(correct_hstack)
        print(f"校正正确/校正总个数: {rio:.4f} ({num} / {len(correct_hstack)})")

        for v in range(n_view):
            hid = hidx[v].cpu()
            c = self.noise_label[v][hid] - self.train_label[v][hid]
            hard.append(c)
        hard_hstack = np.hstack(hard)
        hard_array = np.where(hard_hstack, 0, 1)
        num = np.sum(hard_array)
        rio = num / len(hard_hstack)
        print(f"hard中干净/hard: {rio:.4f} ({num} / {len(hard_hstack)})")

    def __getitem__(self, index):
        if self.prob is None:
            return [self.train_data[v][index] for v in range(len(self.train_data))], [self.noise_label[v][index] for v in range(len(self.train_data))], index
        else:
            return [self.train_data[v][index] for v in range(len(self.train_data))], [self.noise_label[v][index] for v in range(len(self.train_data))], [self.prob[v][index] for v in range(len(self.prob))], index

    def __len__(self):
        return len(self.train_data[0])
