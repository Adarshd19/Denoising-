import os
import sys
import yaml
import time
import cv2
import shutil
import argparse
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from torch import optim
import torch.nn.functional as F
import pkbar
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms,datasets

sys.path.append('..')
from models.unet import UNet
from loss import  mseloss, batch_psnr
from data_loader import get_dataset
from utils import get_logger, save_checkpoint, calc_time, store_images
from utils import  average_meter, weights_init
from utils import get_gpus_memory_info, calc_parameters_count
from models import get_model
from tensorboardX import SummaryWriter

class Network(object):

    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        # self._check_resume()

    def _init_configure(self):
        parser = argparse.ArgumentParser(description='config')

        # Add default argument
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

    def _init_logger(self):
        log_dir = './logs/'+ self.model_name + '/train' + '/{}'.format(self.cfg['data']['dataset']) \
                  +'/{}'.format(time.strftime('%Y%m%d-%H%M%S'))
        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        self.logger.info('{}-Train'.format(self.model_name))
        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.save_image_path = os.path.join(self.save_path, 'saved_val_images')
        self.ckpt_path = os.path.join(self.save_path, 'ckpt')
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.writer = SummaryWriter(self.save_tbx_log)
        shutil.copy(self.args.config, self.save_path)

    def _init_device(self):
        if torch.cuda.is_available():
            self.device_id, self.gpus_info = get_gpus_memory_info()
            self.device = torch.device('cuda:{}'.format(0 if self.cfg['training']['multi_gpus'] else self.device_id))
            print('device: ', self.device)
        else:
            self.logger.info('no gpu device available')
            self.device = 'cpu'

        np.random.seed(self.cfg.get('seed', 1337))
        torch.manual_seed(self.cfg.get('seed', 1337))
        torch.cuda.manual_seed(self.cfg.get('seed', 1337))
        cudnn.enabled = True
        cudnn.benchmark = True



    def _init_dataset(self):
        trainset = get_dataset(mode='train')
        valset = get_dataset(mode ='val')

        print('No. of train images: ', len(trainset))
        print('No. of val images: ', len(valset))

        self.batch_size = self.cfg['training']['batch_size']
        kwargs = {'num_workers': self.cfg['training']['n_workers'], 'pin_memory': True}

        self.train_queue = data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)

        self.valid_queue = data.DataLoader(valset, batch_size=self.batch_size, num_workers=8, drop_last=False, pin_memory=True)


    def _init_model(self):

        criterion = mseloss()
        self.criterion = criterion.to(self.device)
        self.cal_psnr = batch_psnr()

        self.logger.info("Using loss {}".format(self.criterion))

        # Setup Model
        model = get_model(self.model_name)
        # model = model.float()

        # init weight using hekming methods
        # print('*_*'*30)
        # print(model)
        # print('*_*'*30)

        model.apply(weights_init)
        self.logger.info('Initialize the model weights: kaiming_uniform')

        if torch.cuda.device_count() > 1 and self.cfg['training']['multi_gpus']:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)
        
        self.model = model.to(self.device)

    def norm(self, img):
        img = (img-torch.min(img))/(torch.max(img)-torch.min(img))

        return img

    def run(self):

        # self.optimizer = optim.Adagrad(self.model.parameters(), lr=1e-3, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        # optim.SGD(self.model.parameters(), lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        self.global_step = 0
        self.no_epoch = self.cfg['training']['epoch']

        for epoch in range(self.no_epoch):

            self.epoch = epoch
            tbar = tqdm(self.train_queue)
            print('Epoch: %d/%d' % (self.epoch + 1, self.no_epoch))
            self.logger.info('=> Epoch {}'.format(self.epoch))
            # train and search the model
            self.model.train()
            self.kbar = pkbar.Kbar(target=25, width=0)
            self.epoch_loss = 0
            self.trainloss = 0
            self.trainpsnr = 0

            for i, batch in enumerate(tbar):
                # print(batch)
                self.optimizer.zero_grad()

                noisy_imgs = batch[0]
                clean_image = batch[1]

                noisy_imgs = noisy_imgs.to(self.device)#, dtype=torch.float32)
                # mask_type = torch.float32 if net.n_classes == 1 else torch.long
                clean_image = clean_image.to(self.device)#, dtype=torch.float32)
                # print('imgs', imgs)
                clean_pred = self.model(noisy_imgs)
                # print('clean_pred', clean_pred)
                clean_pred = self.norm(clean_pred)
                # clean_pred = noisy_imgs - noise_pred
                # true_masks = noisy_imgs-noise   #clean_imgs

                self.loss = self.criterion(clean_pred, clean_image)
                psnr = self.cal_psnr(clean_pred, clean_image,1.)


                # self.loss, psnr= self.criterion(noise_pred, noise, clean_pred, true_masks)
                # self.loss = self.loss.to(device=self.device)
                self.trainloss += self.loss
                self.trainpsnr += psnr
                print('Loss: ', self.loss.item())
                print('trainpsnr', psnr)
                self.epoch_loss += self.loss.item()
                self.writer.add_scalar('Loss/train', self.loss.item(), self.global_step)

                # pbar.set_postfix(**{'loss (batch)': loss.item()})

                if self.cfg['training']['grad_clip']:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                        self.cfg['training']['grad_clip'])

                # self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                self.kbar.update(i, values=[("loss", self.loss.detach().cpu().numpy())])

            self.trainloss = self.trainloss/(len(self.train_queue))
            self.trainpsnr = self.trainpsnr/(len(self.train_queue))

            self.logger.info('TrainLoss : {}'.format(self.trainloss))
            self.logger.info('Trainpsnr : {}'.format(self.trainpsnr))

            # print(torch.sum(list(self.model.parameters())[0]))
            # for param in self.model.parameters():
            #     print(param.data)

            # valid the model
            print('Starting validation....')

            self.model.eval()
            self.tot = 0
            self.tot_psnr = 0

            self.global_step += 1

            tbar_val = tqdm(self.valid_queue)

            for i, batch in enumerate(tbar_val):

                imgs = batch[0]
                clean_image_val = batch[1]

                imgs = imgs.to(self.device)#, dtype=torch.float32)
                # mask_type = torch.float32 if net.n_classes == 1 else torch.long
                clean_image_val = clean_image_val.to(self.device)#, dtype=torch.float32)

                clean_pred_val = self.model(imgs)
                clean_pred_val = self.norm(clean_pred_val)
                # clean_pred1 = imgs-noise_pred_val
                # clean_img1 = imgs- noise
                self.val_loss = self.criterion(clean_pred_val, clean_image_val)
                self.psnr_val = self.cal_psnr(clean_pred_val, clean_image_val,1.)
                # self.val_loss, self.psnr_val = self.criterion(noise_pred_val, noise, clean_pred1, clean_img1)
                self.tot += self.val_loss#.to(device=self.device)
                self.tot_psnr += self.psnr_val



            self.val_score = self.tot / (len(self.valid_queue))
            self.psnr_score = self.tot_psnr / (len(self.valid_queue))
            self.logger.info('ValidLoss : {}'.format(self.val_score))
            self.logger.info('Validpsnr : {}'.format(self.psnr_score))

            # self.logging.info('Validation Dice Coeff: {}'.format(self.val_score))
            # self.writer.add_scalar('Dice/test', self.val_score, self.global_step)

            self.writer.add_images('images', noisy_imgs, self.global_step) #noisy
            self.writer.add_images('masks/true', clean_image, self.global_step) #clean
            self.writer.add_images('masks/pred', clean_pred, self.global_step) #pred should be close to clean

            # self.kbar.add(i, values=[("val_score", self.val_score)])
            # self.kbar.add(i, values=[("PSNR_val", self.psnr_score)])
            print("val_score", self.val_score)
            print("PSNR_val", self.psnr_score)

            self.filename = 'ckpt_{0}'.format(self.epoch + 1) + '.pth.tar'
            torch.save({
                'epoch': self.epoch + 1,
                'state_dict': self.model.state_dict()
                }, self.filename)
            shutil.move(self.filename, self.ckpt_path)



        # export scalar data to JSON for external processing
        self.writer.close()
        self.logger.info('log dir in : {}'.format(self.save_path))



if __name__ == '__main__':

    train_network = Network()
    train_network.run()