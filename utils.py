import os

import torch
import torch.nn.functional as F

import torch.nn as nn

import numpy as np
import cv2

import os.path as osp
import pickle

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util


def tensor_to_numpy(tensor):
    t_numpy = tensor.cpu().numpy()
    t_numpy = np.transpose(t_numpy, [0, 2, 3, 1])
    t_numpy = np.squeeze(t_numpy)

    return t_numpy

class ToTensor:
    def __call__(self, data):
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        elif len(data.shape) == 3:
            data = data.transpose((2, 0, 1))
        else:
            print("Unsupported shape!")
        return torch.from_numpy(data)

class Normalize:
    def __call__(self, image):
        image = image.astype(np.float32) / 255
        return image

class Horizontal_flip:
    def __call__(self, image):
        # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
        image = image[:, ::-1]
        return image

class Rescale:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        return cv2.resize(image, (self.output_size, self.output_size), cv2.INTER_AREA)

def create_splits(output_dir, image_dir):
    png_files = subfiles(image_dir, suffix=".png", join=False)

    trainset_size = len(png_files)*60//100
    valset_size = len(png_files)*30//100
    testset_size = len(png_files)*10//100

    splits = []
    for split in range(0, 5):
        image_list = png_files.copy()
        trainset = []
        valset = []
        testset = []
        for i in range(0, trainset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            trainset.append(patient[:-4])
        for i in range(0, valset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            valset.append(patient[:-4])
        for i in range(0, testset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            testset.append(patient[:-4])
        split_dict = dict()
        split_dict['train'] = trainset
        split_dict['val'] = valset
        split_dict['test'] = testset

        splits.append(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)

def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

