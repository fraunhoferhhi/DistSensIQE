'''
The copyright in this software is being made available under this Software
Copyright License. This software may be subject to other third party and
contributor rights, including patent rights, and no such rights are
granted under this license.
Copyright (c) 1995 - 2020 Fraunhofer-Gesellschaft zur Förderung der
angewandten Forschung e.V. (Fraunhofer)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted for purpose of testing the functionalities of
this software provided that the following conditions are met:
*     Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
*     Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
*     Neither the names of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.
'''


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

import os
import numpy as np
import random
from sklearn.feature_extraction.image import extract_patches_2d

import matplotlib.pyplot as plt


class IqaDataset(Dataset):

    def __init__(self, iqaDataFrame, transform=None, return_numpy=True):


        """
        Dataset class that wraps around an iqaDataFrame and can be used with 
        an pytorch dataloader. 

        Parameters:
        -----------

        iqaDataFrame : pandas.DataFrame,
            DataFrame as produced by iqaDataFrame that has the following 
            columns: "path_ref", "path_dist", "q_norm", "quality", "dataset", 
            "refname", "distortion", "height", "width", "fps", "mse"

        transform : pytorch transform,
            transform to be applied to each sample

        return_numpy : bool
            If True, images are returned as numpy arrays. If False images are 
            returned as PIL images.

        """

        self.iqaDataFrame = iqaDataFrame
        self.transform = transform
        self.return_numpy = return_numpy


    def __len__(self):
        return self.iqaDataFrame.shape[0]

    def __getitem__(self, idx):

        """
        Returns:
        --------

        sample: dict
            Dictonary holding reference and distorted image as well as meta 
            information.
        """

        # load images
        img_ref = Image.open(
            self.iqaDataFrame.iloc[idx].path_ref).convert("L")
        
        img_dist = Image.open(
            self.iqaDataFrame.iloc[idx].path_dist).convert("L")

        if self.return_numpy:
            # convert PIL -> numpy.ndarray
            img_ref = np.array(img_ref)
            img_dist = np.array(img_dist)
            h, w = img_ref.shape
            img_ref = np.reshape(img_ref, [1, h, w])
            img_dist = np.reshape(img_dist, [1, h, w])

        # sample is a dict
        sample = {"path_ref" : self.iqaDataFrame.iloc[idx].path_ref,
                  "path_dist" : self.iqaDataFrame.iloc[idx].path_dist,
                  "img_ref" : img_ref, 
                  "img_dist" : img_dist, 
                  "q_norm" : torch.Tensor(
                    [self.iqaDataFrame.iloc[idx].q_norm]), 

                  "quality" : torch.Tensor(
                    [self.iqaDataFrame.iloc[idx].quality]),

                  "dataset" : self.iqaDataFrame.iloc[idx].dataset,
                  "refname" : self.iqaDataFrame.iloc[idx].refname,
                  "distortion" : self.iqaDataFrame.iloc[idx].distortion, 
                  "height" : self.iqaDataFrame.iloc[idx].height,
                  "width" : self.iqaDataFrame.iloc[idx].width,
                  "fps" : self.iqaDataFrame.iloc[idx].fps,
                  "mse" : torch.Tensor([self.iqaDataFrame.iloc[idx].mse])
                  }

        if self.transform:
            # apply  transforms
            sample = self.transform(sample)

        return sample


class RandomBlocks(object):

    def __init__(self, block_size, max_blocks):

        """
        Extract co-located blocks from random spatial positions.

        Parameters:
        -----------

        block_size : int,
            Height and width of blocks.

        max_blocks : int,
            Number of blocks to be extracted.
        """

        self.block_size = block_size
        self.max_blocks = max_blocks


    def __call__(self, sample):



        # channels, height, width
        c, h, w = sample["img_ref"].shape
        # channels_ref + channels_dist, height, width
        ref_dist = np.concatenate(
            [sample["img_ref"], sample["img_dist"]], axis=0)

        # height, width, channels_ref + channels_dist
        ref_dist = np.transpose(ref_dist, [1,2,0])

        # blocks, height, width, channels_ref + channels_dist
        blocks = extract_patches_2d(
            ref_dist, patch_size=(self.block_size, self.block_size),  
            max_patches=self.max_blocks)
        
        # blocks, channels_ref + channels_dist, height, width
        blocks = np.transpose(blocks, [0,3,1,2])
        
        # blocks, channels, height, width
        sample["img_ref"]  = blocks[:, :c]
        sample["img_dist"] = blocks[:, c:]

        return sample

        
class ToTensor(object):

    """
    The build-in ToTensor class reduces the pixel range 
    from [0, 255] to [0, 1]. This class keeps the original range.
    """

    def __call__(self, sample):

        sample["img_ref"] = torch.Tensor(sample["img_ref"])

        sample["img_dist"] = torch.Tensor(sample["img_dist"])

        return sample


class FlipHorizontal(object):

    """Flip images horizontally with a probability of 0.5."""

    def __call__(self, sample):

        """
        Parameters:
        -----------

        sample : dict,
            Dictionary holding reference and distorted images. Importantly, 
            images must be tensors not PIL images,

        """

        if random.getrandbits(1):

            # axis -2 is width dimension for both B, H, W, C and H, W, C 
            # formats

            sample["img_ref"] = torch.flip(sample["img_ref"], dims=[-2])
            sample["img_dist"] = torch.flip(sample["img_dist"], dims=[-2])

        return sample

        


if __name__ == "__main__":

    from iqaDataFrame import IqaDataFrame

    # runnung as main is useful for debugging

    datadir = "/home/becker/repositories/iqe/datasets"

    iqedata = IqaDataFrame(
        datadir=datadir, datasets=["liveiqa"], nTrain=17, nVal=6)
    
    # the order of the transforms matters as FlipHorizontalTransform expects 
    # pytorch tensors whereas the other transforms expect numpy arrays
    dataset = IqaDataset(
        iqedata.df_train, datadir=datadir, transform=transforms.Compose(
            [RandomBlocks(320, 1), ToTensor(), FlipHorizontalTransform()]))

    dataloader = DataLoader(dataset, batch_size=4)

    for i_batch, sample_batched in enumerate(dataloader):

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(sample_batched["img_ref"][0][0].squeeze(), "gray")
        ax[1].imshow(sample_batched["img_dist"][0][0].squeeze(), "gray")
        plt.show()

    print("done")
