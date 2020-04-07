import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import cv2
import numpy as np
import sys
from skimage.measure.simple_metrics import compare_psnr#peak_signal_noise_ratio

__all__ = ['SegmentationLosses']


class batch_psnr(nn.Module):
    def __init__(self):
        super(batch_psnr, self).__init__()

    def forward(self, img, imclean, data_range):
        Img = img.data.detach().cpu().numpy().astype(np.float32)
        Iclean = imclean.data.detach().cpu().numpy().astype(np.float32)
        PSNR = 0
        # print(Img.shape)
        for i in range(Img.shape[0]):
            PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
            # print(PSNR)
        return (PSNR/Img.shape[0])  

class mseloss(nn.MSELoss):
    def __init__(self):
        super(mseloss, self).__init__()
        self.mse = nn.MSELoss()

    def batch_mse(self, img, imclean):
        # Img = img.data.cpu().numpy().astype(np.float32)
        # Iclean = imclean.data.cpu().numpy().astype(np.float32)
        loss = 0
        # print(img.shape)
        for i in range(img.size(0)):
            loss += self.mse(imclean[i,:,:,:], img[i,:,:,:])
        return (loss/img.shape[0])

    def forward(self, input1, target):#, clean_pred, clean_img):

        inp = input1 #[i].unsqueeze(0)
        targ = target#[i].unsqueeze(0)

        loss = self.batch_mse(inp,targ)
        

        return loss

