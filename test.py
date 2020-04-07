import os
import cv2
import sys
import yaml
import time
import numpy
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from natsort import natsorted
from PIL import *
from torchsummary import summary
import torch.nn.functional as F

sys.path.append('..')
from models.unet import UNet
from loss import  mseloss, batch_psnr
from data_loader import get_dataset
from utils import get_logger, save_checkpoint, calc_time, store_images
from utils import  average_meter, weights_init
from utils import get_gpus_memory_info, calc_parameters_count
from models import get_model
from tensorboardX import SummaryWriter
from skimage.measure.simple_metrics import compare_psnr #peak_signal_noise_ratio


# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

base_log_path = '/home/mihir/Desktop/Adarsh/otc/logs/unet/train/Data/'
img_base_path = '/home/mihir/Desktop/Adarsh/otc/Data/denoise'

def show(im):
        cv2.imshow('image', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Network(object):

    def __init__(self):
        self._init_configure()
        self._init_device()
        self._init_dataset()
        self._init_model()

    def _init_configure(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--config',nargs='?',type=str,default='./configs/config.yml',
                            help='Configuration file to use')
        parser.add_argument('--model',nargs='?',type=str,default='unet',
                            help='Model to train and evaluation')

        self.args = parser.parse_args()

        with open(self.args.config) as fp:
            self.cfg = yaml.load(fp)
            print('load configure file at {}'.format(self.args.config))
        self.model_name = self.args.model
        print('Usage model :{}'.format(self.model_name))

    def _init_device(self):
        if torch.cuda.is_available():
            self.device_id, self.gpus_info = get_gpus_memory_info()
            self.device = torch.device('cuda:{}'.format(0 if self.cfg['training']['multi_gpus'] else self.device_id))
            print('device: ', self.device)
        else:
            print('no gpu device available')
            self.device = 'cpu'

        np.random.seed(self.cfg.get('seed', 1337))
        torch.manual_seed(self.cfg.get('seed', 1337))
        torch.cuda.manual_seed(self.cfg.get('seed', 1337))
        cudnn.enabled = True
        cudnn.benchmark = True

    def _init_dataset(self):

        class ReadImages(object):

            def __init__(self, path):
                super(ReadImages).__init__()

                self.path = path
                self.images = natsorted(os.listdir(self.path))
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize(mean = [0.321, 0.165, 0.061], std = [0.237, 0.127, 0.051])
                ])

            def __len__(self):
                return len(self.images)

            def norm(self, img):
                img = (img-np.min(img))/(np.max(img)-np.min(img))
                return img

            def __getitem__(self, index):
                img = cv2.imread(os.path.join(self.path, self.images[index]),0)

                img = cv2.resize(img, (256,256))
                img = img.astype('uint8')
                img = self.norm(img)
                img = np.expand_dims(img, axis=-1)
                img = self.transform(img).float()
                return img
        
        img_path = img_base_path  + '/val/_imgs'
        test_images = ReadImages(img_path)
        print('No. of val images: ', len(test_images))
        self.batch_size = self.cfg['training']['batch_size']
        kwargs = {'num_workers': self.cfg['training']['n_workers'], 'pin_memory': True}

        self.valid_queue = data.DataLoader(test_images, batch_size=self.batch_size, num_workers=8, drop_last=False, pin_memory=True)

    def _init_model(self):
        # criterion = mseloss()
        # self.criterion = criterion.to(self.device)
        self.cal_psnr = batch_psnr()

        # print("Using loss {}".format(self.criterion))

        # Setup Model
        model = get_model(self.model_name)
        # model = model.float()

        # init weight using hekming methods
        # print('*_*'*30)
        # print(model)
        # print('*_*'*30)

        if torch.cuda.device_count() > 1 and self.cfg['training']['multi_gpus']:
            model = nn.DataParallel(model)
        else:
            torch.cuda.set_device(self.device_id)
        
        self.model = model.to(self.device)
    
    def t_norm(self, img):
            img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            return img

    def batch_PSNR(self, img, imclean, data_range):
        Img = img.data.cpu().numpy().astype(np.float32)
        Iclean = imclean.data.cpu().numpy().astype(np.float32)
        PSNR = 0
        # print(Img.shape)
        for i in range(Img.shape[0]):
            PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
            # print(PSNR)
        return (PSNR/Img.shape[0])
 
            

    def run(self):
        self.model.load_state_dict(torch.load(base_log_path + '20200313-154356/ckpt/ckpt_10.pth.tar')['state_dict'])

        if os.path.isdir(img_base_path+ '/val/_preds') == False:
            os.mkdir(img_base_path+ '/val/_preds')
        else:
            print('Directory all ready exists')

        input_files = natsorted(os.listdir(img_base_path+ '/val/_imgs'))
        k=0
        with torch.no_grad():
            for j,img in enumerate(tqdm(self.valid_queue)):
                
                img = img.to(self.device)
                # i = img[0].cpu().detach().numpy()
                # i = np.transpose(i, (1,2,0))
                # show(i)
                out = self.model(img)
                out = self.t_norm(out)
                psnr = self.batch_PSNR(img, out, 1.)
                out = out.data.cpu().detach()
                # out = out[0].numpy()
                for i in range(out.shape[0]):
                    Img = out[i,:,:,:]
                    Img = Img.numpy()
                    pred = np.transpose(Img, (1,2,0))
                    pred = pred*255
                    cv2.imwrite(img_base_path+ '/val/_preds/' + input_files[k], pred)
                    k+=1
                
                # cv2.imwrite(img_base_path+ '/val/_preds/' + input_files[k], pred)
                # psnr = batch_PSNR(img, out,1.)
                # print('psnr = ', psnr)

if __name__ == '__main__':

    # base_log_path = '/home/mihir/Desktop/Adarsh/otc/logs/unet/train/Data/'
    train_network = Network()
    train_network.run()
