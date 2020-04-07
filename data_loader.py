from __future__ import print_function
import os
import random
import sys
import numpy as np
# from .base import BaseDataset
from PIL import Image
from natsort import natsorted
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import time
# import torchvision.datasets as dset

dir = '/home/mihir/Desktop/Adarsh/otc/Data'

def show(im):
    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_dataset(root, mode, dirname):
    base_path = os.path.join(root, dirname)
    image_path = base_path + '/' + mode + '/_imgs' 
    mask_path = base_path + '/' + mode + '/_masks'
    
    images = natsorted(os.listdir(image_path))
    images_list = []

    for image_name in images:
        img_path = os.path.join(image_path, image_name)
        # if mode == ['train','val']:
        mask_name = image_name.split('.')[0] + '_mask.png'
        
        img_mask_path = os.path.join(mask_path, mask_name)
        images_list.append((img_path, img_mask_path))

    return images_list

class loader():
    BASE_DIR = 'denoise'
    N_CLASS = 1
    IN_CHANNELS = 1
    CLASS_WEIGHTS = None
    def __init__(self, root,  split='train', mode=None, ft=False):
        super(loader, self).__init__()
        # self.img_path = root+'denoise/train/_imgs'
        # self.target_path = root+'denoise/val/_masks'

        # self.img_list = natsorted(os.listdir(self.img_path))
        # self.target_list = natsorted(os.listdir(self.target_path))
        # self.size = 512
        self.root = os.path.expanduser(root)

        if mode in ['train', 'val']:
            self.data_info = make_dataset(self.root, mode, self.BASE_DIR) # 'data_clean': after clean, before: 'train'
        else:
            self.data_info = make_dataset(self.root, mode, self.BASE_DIR)

        if len(self.data_info) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
            "Supported image extensions are: " + ",".join('tif')))

        # self.root = os.path.expanduser(root)
        
        self.transform = transforms.Compose([
                            # transforms.ToPILImage(),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean = [0.25225686 0.25225686 0.25225686], std = [0.22672841 0.22672841 0.22672841])
                            ])
#mean: [[0.25225686 0.25225686 0.25225686]]; std: [[0.22672841 0.22672841 0.22672841]]

    def patchmaker(self, img, win, stride=1):
        k = 0
        endc = img.shape[0]
        endw = img.shape[1]
        endh = img.shape[2]
        patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
        TotalPatNum = patch.shape[1] * patch.shape[2]
        Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
        for i in range(win):
            for j in range(win):
                patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
                Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
                k = k + 1
        return Y.reshape([endc, win, win, TotalPatNum])

    def norm(self, img):
        img = (img-np.min(img))/(np.max(img)-np.min(img))

        return img


    def __getitem__(self, index):


        img_path, target_path = self.data_info[index][0], self.data_info[index][1]

        mean=0.23416294
        std=0.21152876
        img = cv2.imread(img_path,0)
        img = cv2.resize(img, (256,256))
        img = img.astype('uint8')
        img = self.norm(img)
        img = np.expand_dims(img, axis=-1)
        # img = img.astype('float')
        # print(np.min(img))
        # img = (img-np.mean(img))/np.std(img)
        # print('img',img)
        # sys.exit(1)

        # print('size:', img.shape))
        # show(img)
        
        target = cv2.imread(target_path,0)
        target = cv2.resize(target, (256,256))
        target = target.astype('uint8')
        target = self.norm(target)
        target = np.expand_dims(target, axis=-1)
        # target = target.astype('float')
        # show(target)
        # noise = img - target
        # show(noise)
        # print(img.shape)
        img = self.transform(img).float()
        target = self.transform(target).float()
        # print(img.size())
        # print(img)
        # print(target)
        # sys.exit(1)
       
        # _, target = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        # mask = mask.astype('uint8')
        # target = np.expand_dims(target, axis=-1)
        # target = np.transpose(target, (2,0,1))
        # target[target == 255] = 1
        # target = torch.from_numpy(target).float()
        # print('size:', target.shape)
        
        
        # print('size:', target.size())
        
        # print('size1111:', target.size())
        

        return img, target

    def __len__(self):

        # assert len(self.img_list == len(self.mask_list)
        return len(self.data_info)

def get_dataset(path=dir, **kwargs):
    return loader(root = path, **kwargs)