import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import  DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage, transforms
import numpy as np
from PIL import Image, ImageOps
import os
import random
import math

class DatasetSTB(torch.utils.data.Dataset):
    def __init__(self, data_path, meta_path, rough_path, mode):
        super(DatasetSTB, self).__init__()
        self.img_dir = data_path
        self.label_dir = meta_path
        self.rough_dir = rough_path
        self.mode = mode
        self.crop_h = 512
        self.crop_w = 512
        self.avgPool = nn.AvgPool2d(8)

        self.examples = []
        

        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            file_name = file_name.split(".")[0]
            img_path = self.img_dir + file_name + ".jpg"

            label_img_path = self.label_dir + file_name + ".npy"
            if self.rough_dir:
                rough_img_path = self.rough_dir + file_name + ".npy"

            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            if self.rough_dir:
                example["rough_img_path"] = rough_img_path
            example["img_id"] = file_name
            self.examples.append(example)

        self.num_examples = len(self.examples)
        self.tran=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        label_img_path = example["label_img_path"]
        label_img = np.load(label_img_path)
        if self.rough_dir:
            rough_img_path = example["rough_img_path"]
            rough_img = np.load(rough_img_path)
        
        img_w = img.size[0]
        img_h = img.size[1]
            
        if self.mode == "train":
            hflip = random.random()
            if (hflip < 0.5):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label_img = np.fliplr(label_img)
                if self.rough_dir:
                    rough_img = np.fliplr(rough_img)
            
            if img_w < self.crop_w:
                res = self.crop_w - img_w
                img = ImageOps.expand(img, border=(math.floor(res/2), 0, math.ceil(res/2), 0), fill=0)
                label_img = np.pad(label_img, ((0,0),(math.floor(res/2),math.ceil(res/2))), 'constant')
                if self.rough_dir:
                    rough_img = np.pad(rough_img, ((0,0),(math.floor(res/2),math.ceil(res/2))), 'constant')
                img_w = self.crop_w

            if img_h < self.crop_h:
                res = self.crop_h - img_h
                img = ImageOps.expand(img, border=(0, math.floor(res/2), 0, math.ceil(res/2)), fill=0)
                label_img = np.pad(label_img, ((math.floor(res/2),math.ceil(res/2)),(0,0)), 'constant')
                if self.rough_dir:
                    rough_img = np.pad(rough_img, ((math.floor(res/2),math.ceil(res/2)),(0,0)), 'constant')
                img_h = self.crop_h

            dx = random.randint(0, img_w-self.crop_w)
            dy = random.randint(0, img_h-self.crop_h)
            img = img.crop((dx, dy, self.crop_w+dx, self.crop_h+dy))
            label_img = label_img[dy:dy+self.crop_h, dx:dx+self.crop_w]
            if self.rough_dir:
                rough_img = rough_img[dy:dy+self.crop_h, dx:dx+self.crop_w]

        img = self.tran(img)
        label_img = label_img.reshape(1, label_img.shape[0], label_img.shape[1])
        label_img = torch.from_numpy(label_img.copy()).float()
        if self.mode == "train":
            label_img = self.avgPool(label_img) * 64
        if self.rough_dir:
            rough_img = rough_img.reshape(1, rough_img.shape[0], rough_img.shape[1])
            rough_img = torch.from_numpy(rough_img.copy()).float()
            rough_img = self.avgPool(rough_img) * 64
        

            rough_img = 2 - rough_img * 15
            rough_img[rough_img < -1.0] = -1.0
            rough_img[rough_img > 2.0] = 2.0
            dilation_map = torch.zeros(18, rough_img.shape[1], rough_img.shape[2])
            dilation_map[0][:][:] = -rough_img[0][:][:] # left top |
            dilation_map[1][:][:] = -rough_img[0][:][:] # left top -
            dilation_map[2][:][:] = -rough_img[0][:][:] # top |
            #dilation_map[3][:][:] = 0.0               # top -
            dilation_map[4][:][:] = -rough_img[0][:][:] # right top |
            dilation_map[5][:][:] = rough_img[0][:][:]  # right top -
            #dilation_map[6][:][:] = 0.0               # left |
            dilation_map[7][:][:] = -rough_img[0][:][:] # left -
            #dilation_map[8][:][:] = 0.0               # middle |
            #dilation_map[9][:][:] = 0.0               # middle -
            #dilation_map[10][:][:] = 0.0              # right | 
            dilation_map[11][:][:] = rough_img[0][:][:] # right -
            dilation_map[12][:][:] = rough_img[0][:][:] # left bottom |
            dilation_map[13][:][:] = -rough_img[0][:][:]# left bottom -
            dilation_map[14][:][:] = rough_img[0][:][:] # bottom |
            #dilation_map[15][:][:] = 0.0              # bottom -
            dilation_map[16][:][:] = rough_img[0][:][:] # right bottom |
            dilation_map[17][:][:] = rough_img[0][:][:] # right bottom -
            
            rough_img = dilation_map

            return img, label_img, rough_img, example["img_id"]
        else:
            return img, label_img, example["img_id"]

    def __len__(self):
        return self.num_examples