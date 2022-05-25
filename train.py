import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch import Tensor
import os
import datetime
from tensorboardX import SummaryWriter
import time
import sys, time
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from PIL import Image
import math

class Trainer(object):
    def __init__(self, optim, scheduler, model, log_dir, device, rough=False):
        self.NAME = "PANet"
        self.model = model
        self.cuda = torch.cuda.is_available()
        self.log_dir = log_dir
        self.optim = optim
        self.epoch = 0
        self.scheduler = scheduler
        self.device = device
        self.rough = rough
        self.model = self.model.to(self.device)
        self.avgPool = nn.AvgPool2d(kernel_size=(8, 8))
        
    def train(self, train_loader, test_loader, loss_function, num_epochs):
        writer = SummaryWriter(log_dir=self.log_dir)
        min_l1loss = 1000.0
        min_l2loss = 1000.0
            
        for epoch in range(self.epoch, num_epochs):
            since = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            print("lr: ", self.optim_g.param_groups[0]['lr'])
            running_loss = 0.0
            total = 0
                
            for i, data in enumerate(train_loader, 0):
                if self.rough:
                    inputs, labels, _ = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                else:
                    inputs, labels, roughs, _ = data
                    inputs, labels, roughs = inputs.to(self.device), labels.to(self.device), roughs.to(self.device)
                

                self.model.train()
                self.optim.zero_grad()

                if self.rough:
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs, roughs)
                
                loss = loss_function(outputs, labels)
                loss.backward()

                self.optim.step()

                running_loss += loss.item() * inputs.size(0)
                
                total += inputs.size(0)
                    
                
            if self.scheduler:
                self.scheduler.step()
           
            epoch_loss = running_loss / total           
            print('Training Loss: {:.4f}'.format(epoch_loss))
            writer.add_scalar('Training_loss', epoch_loss, epoch)
           
            
            # Test
            self.model.eval()
            running_l1loss = 0.0
            running_l2loss = 0.0
            total = 0
            for i, data in enumerate(test_loader, 0):
                if self.rough:
                    inputs, labels, _ = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                else:
                    inputs, labels, roughs, _ = data
                    inputs, labels, roughs = inputs.to(self.device), labels.to(self.device), roughs.to(self.device)
                
                    
                
                with torch.set_grad_enabled(False):
                    if self.rough:
                        outputs = self.model(inputs)
                    else:  
                        outputs = self.model(inputs, roughs)
                        
                    lossl1 = torch.abs(torch.sum(outputs) - torch.sum(labels))
                    lossl2 = (torch.sum(outputs) - torch.sum(dots)).square()

                total += inputs.size(0)
                running_l1loss += lossl1.item()
                running_l2loss += lossl2.item()
            
            epoch_l1loss = running_l1loss / total
            epoch_l2loss = math.sqrt(running_l2loss / total)
            print('Test L1 Loss: {:.4f}'.format(epoch_l1loss))
            print('Test L2 Loss: {:.4f}'.format(epoch_l2loss))
            writer.add_scalar('Test_l1_loss', epoch_l1loss, epoch)
            writer.add_scalar('Test_l2_loss', epoch_l2loss, epoch)
            
            
            time_elapsed = time.time() - since
            print('one epoch complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            if epoch_l1loss < min_l1loss or epoch_l2loss < min_l2loss:
                min_l1loss = min(min_l1loss, epoch_l1loss)
                min_l2loss = min(min_l2loss, epoch_l2loss)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'epoch': epoch,
                }, self.log_dir+'epoch{:04d}.pt'.format(epoch))
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'epoch': epoch,
                }, self.log_dir+"best.pt")
            

        writer.close()
        print("train finished")

    def load_weights(self, path, epoch=-1, inference=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not inference:
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            if epoch == -1:
                self.epoch = checkpoint['epoch']
            else:
                self.epoch = epoch
        print("load weights from {} finished.".format(path))
    
    def evaluate(self, val_loader):
        self.model = self.model.eval()
        with torch.no_grad():
            running_l1loss = 0.0
            running_l2loss = 0.0
            total = 0
            for i, data in enumerate(val_loader, 0):
                if self.rough:
                    inputs, labels, img_id = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    preds = self.model(inputs)
                else:
                    inputs, labels, roughs, img_id = data
                    inputs, labels, roughs = inputs.to(self.device), labels.to(self.device), roughs.to(self.device)
                    preds = self.model(inputs, roughs)
                
                
                l1loss = torch.abs(torch.sum(preds) - torch.sum(labels))
                l2loss = (torch.sum(preds) - torch.sum(labels)).square()

                total += inputs.size(0)
                running_l1loss += l1loss.item()
                running_l2loss += l2loss.item()
            
            l1loss = running_l1loss / total
            l2loss = math.sqrt(running_l2loss / total)
        print("l1 loss: {:.4f}".format(l1loss))
        print("l2 loss: {:.4f}".format(l2loss))

    def rough_save(self, test_loader, save_dir):
        self.model = self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                if self.rough:
                    inputs, labels, img_id = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    preds = self.model(inputs)
                else:
                    inputs, labels, roughs, img_id = data
                    inputs, labels, roughs = inputs.to(self.device), labels.to(self.device), roughs.to(self.device)
                    preds = self.model(inputs, roughs)
                    
                nums = torch.sum(preds)
                preds = F.interpolate(preds, size=(inputs.shape[2], inputs.shape[3]), mode="bilinear", align_corners=True)

                preds = preds / torch.sum(preds) * nums
                
                preds = preds.detach().squeeze().cpu().numpy()
                np.save(save_dir + img_id[0] + ".npy", preds)

    def target_save(self, test_loader, save_dir):
        self.model = self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                if self.rough:
                    inputs, labels, img_id = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    preds = self.model(inputs)
                else:
                    inputs, labels, roughs, img_id = data
                    inputs, labels, roughs = inputs.to(self.device), labels.to(self.device), roughs.to(self.device)
                    preds = self.model(inputs, roughs)
                    
                nums = torch.sum(labels) # true nums in GT
                preds = F.interpolate(preds, size=(inputs.shape[2], inputs.shape[3]), mode="bilinear", align_corners=True)

                preds = preds / torch.sum(preds) * nums
                
                preds = preds.detach().squeeze().cpu().numpy()
                np.save(save_dir + img_id[0] + ".npy", preds)
