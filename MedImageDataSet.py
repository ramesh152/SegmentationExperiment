#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import fnmatch
import random

import numpy as np

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading import MultiThreadedAugmenter

def get_transforms(mode="train", target_size=128):
    tranform_list = []

    if mode == "train":
        tranform_list = [# CenterCropTransform(crop_size=target_size),
                         ResizeTransform(target_size=(target_size,target_size), order=1),
                         MirrorTransform(axes=(1,)),
                         ]


    elif mode == "val":
        tranform_list = [CenterCropTransform(crop_size=target_size),
                         ResizeTransform(target_size=target_size, order=1),
                         ]

    elif mode == "test":
        tranform_list = [CenterCropTransform(crop_size=target_size),
                         ResizeTransform(target_size=target_size, order=1),
                         ]

    tranform_list.append(NumpyToTensor())

    return Compose(tranform_list)

def load_data_set(root_dir,mode=None,keys=None,taskname=None):
    image_names = keys

    dataDir = "imagesTr"
    maskDir = "masksTr"
    suffix =".png"  
    img_data = []
    img_labels = []

    for image in self.image_names : 
        img = cv2.imread(osp.join(root_dir,taskname,dataDir,image+suffix))
        print("image path: ",osp.join(self.root_dir,self.taskname,dataDir,img+suffix))
        img_data.append(img)
                
        target_img = np.zeros(img.shape[:2], dtype=np.uint8)
        target_img_ = cv2.imread(osp.join(root_dir,taskname,maskDir,image+suffix),0)
        target_img = np.maximum(target_img, target_img_)
        img_labels.append(target_img)

    return img_data,img_labels

class MedImageDataSet(object):
     """
       TODO
     """
     def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000, seed=None, num_processes=8,    num_cached_per_queue=8 * 4, target_size=128, file_pattern='*.png', do_reshuffle=True, keys=None):

        data_loader = MedImageDataLoader(base_dir=base_dir, mode=mode, batch_size=batch_size, 
        num_batches=num_batches, seed=seed,file_pattern=file_pattern,keys=keys)

        self.data_loader = data_loader
        self.batch_size = batch_size
        #self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1

        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedAugmenter(data_loader, self.transforms, num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue, seeds=seed,
                                                 shuffle=do_reshuffle)
        self.augmenter.restart()

     def __len__(self):
        return len(self.data_loader)

     def __iter__(self):
        self.augmenter.renew()
        return self.augmenter

     def __next__(self):
        return next(self.augmenter)


class MedImageDataLoader(DataLoader):
     def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234, return_incomplete=False,
                 shuffle=True, infinite=False):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        self.img_data ,self.img_labels = load_data_set(self.root_dir,self.mode,self.image_names,self.taskname)
        self.data = list(range(len(self.img_data)))

        super().__init__(self.data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.patch_size = patch_size
        self.batch_size = batch_size

        self.use_next = False
        if mode == "train":
            self.use_next = False

        self.indices = list(range(len(self.img_data))) 
        self.data_len = len(self.img_data)

        self.num_batches = min((self.data_len // self.batch_size)+10, num_batches)

     def generate_train_batch(self):
        
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        
        data = []
        labels = []
        
        for idx in patients_for_batch:
           data.append(self.img_data[idx])
           labels.append(self.img_labels[idx])
        return {'data': data, 'seg':seg, 'metadata':metadata, 'names':patient_names}

     def __len__(self):
        n_items = min(self.data_len // self.batch_size, self.num_batches)
        return n_items


      
        


       
 

