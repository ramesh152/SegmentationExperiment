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
import pickle
from collections import OrderedDict
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from networks.UNET import UNet
from NucleusDataset import NucleusDataset
from trixi.experiment.pytorchexperiment import PytorchExperiment
from torchvision import transforms
from utils import tensor_to_numpy,ToTensor,Normalize,Rescale,create_splits
from loss import calc_loss,soft_dice


class UNetExperiment(PytorchExperiment):
    """
    The UnetExperiment is inherited from the PytorchExperiment. It implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    It is optimized to work with the provided NumpyDataLoader.

    The basic life cycle of a UnetExperiment is the same s PytorchExperiment:

        setup()
        (--> Automatically restore values if a previous checkpoint is given)
        prepare()

        for epoch in n_epochs:
            train()
            validate()
            (--> save current checkpoint)

        end()
    """

    def setup(self):
        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)
  
        tr_keys = splits[self.config.fold]['train']
        val_keys = splits[self.config.fold]['val']
        test_keys = splits[self.config.fold]['test']
        print("pkl_dir: ",pkl_dir) 
        print("tr_keys: ",tr_keys)
        print("val_keys: ",val_keys)
        print("test_keys: ",test_keys)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        task = self.config.dataset_name
        self.train_data_loader = torch.utils.data.DataLoader(
        NucleusDataset(self.config.data_root_dir, train=True,
                       transform=transforms.Compose([
                           Normalize(),
                           Rescale(self.config.patch_size),
                           ToTensor()
                       ]),
                       target_transform=transforms.Compose([
                           Normalize(),
                           Rescale(self.config.patch_size),
                           ToTensor()
                       ]),
                      mode ="train",
                      keys = tr_keys,
                      taskname = task),
        batch_size=self.config.batch_size, shuffle=True)

        self.val_data_loader = torch.utils.data.DataLoader(
        	      NucleusDataset(self.config.data_root_dir, train=True,
                   				    transform=transforms.Compose([
                      						     Normalize(),
                          					     Rescale(self.config.patch_size),
                          					     ToTensor() ]),
                                                    target_transform=transforms.Compose([
                                                                     Normalize(),
                                                                     Rescale(self.config.patch_size),
                                                                     ToTensor()]),
                      mode ="val",
                      keys = val_keys,
                      taskname = self.config.dataset_name),
        batch_size=self.config.batch_size, shuffle=True)

        self.test_data_loader = torch.utils.data.DataLoader(
        NucleusDataset(self.config.data_root_dir, train=True,
                       transform=transforms.Compose([
                           Normalize(),
                           Rescale(self.config.patch_size),
                           ToTensor()
                       ]),
                       target_transform=transforms.Compose([
                           Normalize(),
                           Rescale(self.config.patch_size),
                           ToTensor()
                       ]),
                      mode ="test",
                      keys = test_keys,
                      taskname = self.config.dataset_name),
        batch_size=self.config.batch_size, shuffle=True)

        #self.model = UNet(num_classes=self.config.num_classes, in_channels=self.config.in_channels)
        self.model = UNet()
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model"))

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        self.model.train()

        data = None
        batch_counter = 0
        metrics = defaultdict(float)
        #running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(self.train_data_loader):
            data, target = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()

            #print("data  shape :",data.shape, "target shape :",target.shape)
            pred = self.model(data)

            #pred_softmax = F.softmax(pred, dim=1) 
            #We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
            #print("pred_softmax  shape :",pred_softmax.shape, "target shape :",target.shape)
            #loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
            loss = F.binary_cross_entropy(pred, target) + soft_dice(pred,target)

            #loss,_ = calc_loss(pred, target, metrics)
            loss.backward()
            self.optimizer.step()

            #running_loss+=loss.item()
            #epoch_loss = running_loss/len(train_data_loader)

            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Epoch: {0} Loss: {1:.4f}'.format(self._epoch_idx, loss.item()))

                #self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch + (batch_counter / self.train_data_loader.num_batches))
                self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch)  
                self.clog.show_image_grid(data.float().cpu(), name="data", normalize=True, scale_each=True, n_iter=epoch)
                self.clog.show_image_grid(target.float().cpu(), name="mask", title="Mask", n_iter=epoch)
                self.clog.show_image_grid(torch.argmax(pred.cpu(), dim=1, keepdim=True), name="unt_argmax", title="Unet", n_iter=epoch)
                #self.clog.show_image_grid(pred.cpu()[:, 1:2, ], name="unt", normalize=True, scale_each=True, n_iter=epoch)

            batch_counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'

    def validate(self, epoch):
        self.elog.print('VALIDATE')
        self.model.eval()

        data = None
        loss_list = []
        acc_list = []
        metrics = defaultdict(float)
        with torch.no_grad():
             for batch_idx, (images, masks) in enumerate(self.val_data_loader):
                data, target = images.to(self.device), masks.to(self.device)
                pred = self.model(data)
                # pred_softmax = F.softmax(pred, dim=1)  
                # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
                # Ramesh check if soft max is needed
                # loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
                # loss = F.binary_cross_entropy(pred, masks)
                
                #loss,dice = calc_loss(pred, target, metrics)
                acc = soft_dice(pred,target) 
                acc_list.append(acc.item())

                loss = F.binary_cross_entropy(pred, target) + soft_dice(pred,target)
                loss_list.append(loss.item())
                
        assert data is not None, 'data is None. Please check if your dataloader works properly'
        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Mean Loss: %.4f Mean Dice :' % (self._epoch_idx, np.mean(loss_list)),np.mean(acc_list))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)
        self.add_result(value=np.mean(acc_list), name='Val_Mean_Accuracy', tag='Accuracy', counter=epoch+1)

        self.clog.show_image_grid(data.float().cpu(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        self.clog.show_image_grid(target.float().cpu(), name="mask_val", title="Mask", n_iter=epoch)
        self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax_val", title="Unet", n_iter=epoch)
        #self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_val", normalize=True, scale_each=True, n_iter=epoch)

    def test(self):
        # TODO
        print('TODO: Implement your test() method here')
