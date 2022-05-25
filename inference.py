from shanghaiTechA import DatasetSTA
from shanghaiTechB import DatasetSTB
from qnrf import DatasetQNRF
from cc50 import DatasetCC50
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from train import Trainer
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model.PANet import CSRNet, PANet
import numpy as np
import os

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % args.cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precise_net = PANet()
    
    data_dir = args.data_dir
    if args.dataset == 'SHHA':
        test_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/test_data/images/", 
            meta_path=data_dir+"ShanghaiTech/part_A_final/test_data/densitymaps_ada/",
            rough_path=args.dilation_path, # pre-inferenced with rough network
            mode="test")
            

    if args.dataset == 'SHHB':
        test_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/test_data/images/", 
            meta_path=data_dir+"ShanghaiTech/part_B_final/test_data/densitymaps_15/",
            rough_path=args.dilation_path, # pre-inferenced with rough network
            mode="test")
           
    if args.dataset == 'QNRF':
        test_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Test_img_resize/", 
            meta_path=data_dir+"QNRF/Test_density15_resize/",
            rough_path=args.dilation_path, # pre-inferenced with rough network
            mode="test")

    if args.dataset == 'CC50':
        test_dataset = DatasetCC50(data_path=data_dir+"CC50/img/test/", 
            meta_path=data_dir+"CC50/density15/test/",
            rough_path=args.dilation_path, # pre-inferenced with rough network
            mode="test")
            
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)
    
    trainer = Trainer(
        optim = None,
        scheduler = None, 
        model = precise_net,
        log_dir = None,
        device = device,
        rough = False)
    trainer.load_weights(args.checkpoint, inference=True)
    trainer.evaluate(test_loader)

if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='SHHA')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--dilation-path', type=str)
    args = parser.parse_args()
    main(args)

