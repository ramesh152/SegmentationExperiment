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
        
        datafldr = "images"
        maskfldr ="masks"

        train_dir = "train"
        val_dir = "val"
        test_dir = "test"
 
        suffix =".png" 
        print("root_dir :",root_dir, " taskname : ",taskname,"self.mode :",self.mode)
        #print(" path : ",osp.join(self.root_dir, taskname))
        
        #if not self._check_task_exists():
        #    raise RuntimeError("Task does not exist")
            
        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        if self.mode=="train":
            print("root_dir :",str(root_dir))
            self.image_names = os.listdir(os.path.join(self.root_dir, train_dir))
            print("train image_names :",self.image_names)
            self.train_data = []
            self.train_labels = []
           
 
            for image_name in tqdm(self.image_names):
                train_img_ = cv2.imread(osp.join(self.root_dir, train_dir, image_name, "images", image_name + ".png"),0)
                train_img = np.zeros(train_img_.shape[:2], dtype=np.uint8) 
                train_img = np.maximum(train_img, train_img_)
                self.train_data.append(train_img)

                target_img = np.zeros(train_img.shape[:2], dtype=np.uint8)
                for target in glob(osp.join(self.root_dir,train_dir, image_name, "masks", "*.png")):
                    target_img_ = cv2.imread(target, 0)
                    target_img = np.maximum(target_img, target_img_)

                self.train_labels.append(target_img)

                
        elif self.mode =="val":

            print("root_dir :",str(root_dir))
            self.image_names = os.listdir(os.path.join(self.root_dir, val_dir))
            print("val image_names :",self.image_names)
            self.val_data = []
            self.val_labels = []

            for image_name in tqdm(self.image_names):
                val_img_ = cv2.imread(osp.join(self.root_dir, val_dir, image_name, "images", image_name + ".png"),0)
                val_img = np.zeros(val_img_.shape[:2], dtype=np.uint8) 
                val_img = np.maximum(val_img, val_img_)
                self.val_data.append(val_img)

                target_img = np.zeros(val_img.shape[:2], dtype=np.uint8)
                for target in glob(osp.join(self.root_dir,val_dir, image_name, "masks", "*.png")):
                    target_img_ = cv2.imread(target, 0)
                    target_img = np.maximum(target_img, target_img_)

                self.val_labels.append(target_img)

        else :
             print("root_dir :",str(root_dir))
             self.image_names = os.listdir(os.path.join(self.root_dir, test_dir))
             print("test image_names :",self.image_names)
             self.test_data = []
             self.test_labels = []

             for image_name in tqdm(self.image_names):
                test_img_ = cv2.imread(osp.join(self.root_dir, test_dir, image_name, "images", image_name + ".png"),0)
                test_img = np.zeros(test_img_.shape[:2], dtype=np.uint8) 
                test_img = np.maximum(test_img, test_img_)
                self.test_data.append(test_img)

                target_img = np.zeros(test_img.shape[:2], dtype=np.uint8)
                for target in glob(osp.join(self.root_dir,test_dir, image_name, "masks", "*.png")):
                    target_img_ = cv2.imread(target, 0)
                    target_img = np.maximum(target_img, target_img_)

                self.test_labels.append(target_img) 

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
