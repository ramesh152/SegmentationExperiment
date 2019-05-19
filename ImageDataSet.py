from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import torch.nn as nn

import numpy as np
import cv2

import os.path as osp
from glob import glob
from tqdm import tqdm


def load_data_set(root_dir,taskname,mode="train",keys=None):
   image_names = keys

   dataDir = "imagesTr"
   maskDir = "masksTr"
   suffix =".png"  
   img_data = []
   img_labels = []

   for image in image_names : 
        img = cv2.imread(osp.join(root_dir,taskname,dataDir,image+suffix))
        #print("image path: ",osp.join(root_dir,taskname,dataDir,image+suffix))
        img_data.append(img)
                
        target_img = np.zeros(img.shape[:2], dtype=np.uint8)
        target_img_ = cv2.imread(osp.join(root_dir,taskname,maskDir,image+suffix),0)
        target_img = np.maximum(target_img, target_img_)
        img_labels.append(target_img)

   return img_data,img_labels

class NucleusDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, target_transform=None, mode ="train",
                  do_reshuffle=True, keys=None,taskname = None,batch_size=16, num_batches=10000000, seed=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.taskname = taskname
        self.image_names = keys
        self.mode = mode
        self.data_len = len(self.image_names)
        self.batch_size = batch_size
        self.num_batches = min((self.data_len // self.batch_size)+10, num_batches)

        dataDir = "imagesTr"
        maskDir = "masksTr"
        suffix =".png" 
        print("root_dir :",root_dir, " taskname : ",taskname,"self.mode :",self.mode)
        print(" path : ",osp.join(self.root_dir, taskname))
        
        if not self._check_task_exists():
            raise RuntimeError("Task does not exist")
            

        if self.mode=="train":

            print(" Mode : ",mode , " train image_names :",self.image_names)
            self.train_data ,self.train_labels =   load_data_set(root_dir=self.root_dir,taskname=self.taskname,mode=self.mode,keys=self.image_names)
            
        elif self.mode =="val":

            print(" Mode : ",mode , " val image_names :",self.image_names)
            self.val_data ,self.val_labels = load_data_set(root_dir=self.root_dir,taskname=self.taskname,mode=self.mode,keys=self.image_names)

        else :

            print(" Mode : ",mode , " test image_names :",self.image_names)
            self.test_data ,self.test_labels = load_data_set(root_dir=self.root_dir,taskname=self.taskname,mode=self.mode,keys=self.image_names)
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        if self.mode=="train":
            image, mask = self.train_data[item], self.train_labels[item]

            if self.transform:
                image = self.transform(image)

            if self.target_transform:
                mask = self.target_transform(mask)

            return image, mask
                                              
        elif self.mode=="val":
            image, mask = self.val_data[item], self.val_labels[item]

            if self.transform:
                image = self.transform(image)

            if self.target_transform:
                mask = self.target_transform(mask)

            return image, mask     
                                              
        else:
            image, mask = self.test_data[item], self.test_labels[item]

            if self.transform:
                image = self.transform(image)

            if self.target_transform:
                mask = self.target_transform(mask)

            return image, mask     

    def _check_exists(self):
        return osp.exists(osp.join(self.root_dir, "train")) and osp.exists(osp.join(self.root_dir, "test"))
    
    def _check_task_exists(self):
        return osp.exists(osp.join(self.root_dir, self.taskname))
