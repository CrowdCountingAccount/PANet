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
    rough_net = CSRNet()
    precise_net = PANet()
    
    data_dir = args.data_dir
    if args.dataset == 'SHHA':
        if args.stage == 'rough':
            train_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/train_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_A_final/train_data/densitymaps_50/",
                rough_path = None,
                mode="train")
            test_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/test_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_A_final/test_data/densitymaps_50/", 
                rough_path = None,
                mode="test")
            tool_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/train_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_A_final/train_data/densitymaps_50/", 
                rough_path = None,
                mode="test") # for rough_save
        elif args.stage == 'teacher':
            train_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/train_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_A_final/train_data/densitymaps_ada/",
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="train")
            test_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/test_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_A_final/test_data/densitymaps_ada/", 
                rough_path='./outputs/'+args.dataset+'/test/rough/',
                mode="test")
            tool_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/train_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_A_final/train_data/densitymaps_ada/", 
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="test") # for target_save
        elif args.stage == 'student':
            train_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/train_data/images/", 
                meta_path='./outputs/'+args.dataset+'/train/target/',
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="train")
            test_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/test_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_A_final/test_data/densitymaps_15/", # no new target on test set
                rough_path='./outputs/'+args.dataset+'/test/rough/',
                mode="test")
            tool_dataset = DatasetSTA(data_path=data_dir+"ShanghaiTech/part_A_final/train_data/images/", 
                meta_path='./outputs/'+args.dataset+'/train/target/',
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="test") # no use in this stage

    if args.dataset == 'SHHB':
        if args.stage == 'rough':
            train_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/train_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_B_final/train_data/densitymaps_50/",
                rough_path = None,
                mode="train")
            test_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/test_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_B_final/test_data/densitymaps_50/", 
                rough_path = None,
                mode="test")
            tool_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/train_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_B_final/train_data/densitymaps_50/", 
                rough_path = None,
                mode="test") # for rough_save
        elif args.stage == 'teacher':
            train_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/train_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_B_final/train_data/densitymaps_15/",
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="train")
            test_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/test_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_B_final/test_data/densitymaps_15/", 
                rough_path='./outputs/'+args.dataset+'/test/rough/',
                mode="test")
            tool_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/train_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_B_final/train_data/densitymaps_15/", 
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="test") # for target_save
        elif args.stage == 'student':
            train_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/train_data/images/", 
                meta_path='./outputs/'+args.dataset+'/train/target/',
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="train")
            test_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/test_data/images/", 
                meta_path=data_dir+"ShanghaiTech/part_B_final/test_data/densitymaps_15/", # no new target on test set
                rough_path='./outputs/'+args.dataset+'/test/rough/',
                mode="test")
            tool_dataset = DatasetSTB(data_path=data_dir+"ShanghaiTech/part_B_final/train_data/images/", 
                meta_path='./outputs/'+args.dataset+'/train/target/',
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="test") # no use in this stage

    if args.dataset == 'QNRF':
        if args.stage == 'rough':
            train_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Train_img_resize/", 
                meta_path=data_dir+"QNRF/Train_density50_resize/",
                rough_path = None,
                mode="train")
            test_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Test_img_resize/", 
                meta_path=data_dir+"QNRF/Test_density50_resize/",
                rough_path = None,
                mode="test")
            tool_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Train_img_resize/", 
                meta_path=data_dir+"QNRF/Train_density50_resize/",
                rough_path = None,
                mode="test") # for rough_save
        elif args.stage == 'teacher':
            train_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Train_img_resize/", 
                meta_path=data_dir+"QNRF/Train_density15_resize/",
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="train")
            test_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Test_img_resize/", 
                meta_path=data_dir+"QNRF/Test_density15_resize/",
                rough_path='./outputs/'+args.dataset+'/test/rough/',
                mode="test")
            tool_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Train_img_resize/", 
                meta_path=data_dir+"QNRF/Train_density15_resize/",
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="test") # for target_save
        elif args.stage == 'student':
            train_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Train_img_resize/", 
                meta_path='./outputs/'+args.dataset+'/train/target/',
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="train")
            test_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Test_img_resize/", 
                meta_path=data_dir+"QNRF/Test_density15_resize/", # no new target on test set
                rough_path='./outputs/'+args.dataset+'/test/rough/',
                mode="test")
            tool_dataset = DatasetQNRF(data_path=data_dir+"QNRF/Train_img_resize/", 
                meta_path='./outputs/'+args.dataset+'/train/target/',
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="test") # no use in this stage

    if args.dataset == 'CC50': # only 1-fold. Change images in the data directory for other folds.
        if args.stage == 'rough':
            train_dataset = DatasetCC50(data_path=data_dir+"CC50/img/train/", 
                meta_path=data_dir+"CC50/density50/train/",
                rough_path = None,
                mode="train")
            test_dataset = DatasetCC50(data_path=data_dir+"CC50/img/test/", 
                meta_path=data_dir+"CC50/density50/test/",
                rough_path = None,
                mode="test")
            tool_dataset = DatasetCC50(data_path=data_dir+"CC50/img/train/", 
                meta_path=data_dir+"CC50/density50/train/",
                rough_path = None,
                mode="test") # for rough_save
        elif args.stage == 'teacher':
            train_dataset = DatasetCC50(data_path=data_dir+"CC50/img/train/", 
                meta_path=data_dir+"CC50/density15/train/",
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="train")
            test_dataset = DatasetCC50(data_path=data_dir+"CC50/img/test/", 
                meta_path=data_dir+"CC50/density15/test/",
                rough_path='./outputs/'+args.dataset+'/test/rough/',
                mode="test")
            tool_dataset = DatasetCC50(data_path=data_dir+"CC50/img/train/", 
                meta_path=data_dir+"CC50/density15/train/",
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="test") # for target_save
        elif args.stage == 'student':
            train_dataset = DatasetCC50data_path=data_dir+"CC50/img/train/", 
                meta_path='./outputs/'+args.dataset+'/train/target/',
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="train")
            test_dataset = DatasetCC50(data_path=data_dir+"CC50/img/test/", 
                meta_path=data_dir+"CC50/density15/test/", # no new target on test set
                rough_path='./outputs/'+args.dataset+'/test/rough/',
                mode="test")
            tool_dataset = DatasetCC50(data_path=data_dir+"CC50/img/train/", 
                meta_path='./outputs/'+args.dataset+'/train/target/',
                rough_path='./outputs/'+args.dataset+'/train/rough/',
                mode="test") # no use in this stage


    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    tool_loader = DataLoader(dataset=tool_dataset, batch_size=1, shuffle=False, drop_last=False)

    criterion = nn.L1Loss()
    
    optimizer = optim.Adam(rough_net.parameters(), 1e-5, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    
    exp_lr_scheduler = None
    if args.stage == 'rough':
        trainer = Trainer(
            optim = optimizer,
            scheduler = exp_lr_scheduler, 
            model = rough_net,
            log_dir = './log/%s/rough/' % (args.dataset)
            device = device, 
            rough = True)
        trainer.train(train_loader, test_loader, criterion, args.max_epoch)
        trainer.load_weights('./log/%s/rough/best.pt' % (args.dataset))
        trainer.rough_save(tool_loader, './outputs/'+args.dataset+'/train/rough/')
        trainer.rough_save(test_loader, './outputs/'+args.dataset+'/test/rough/')
    elif args.stage == 'teacher':
        trainer = Trainer(
            optim = optimizer,
            scheduler = exp_lr_scheduler, 
            model = precise_net,
            log_dir = './log/%s/teacher/' % (args.dataset)
            device = device, 
            rough = False)
        trainer.train(train_loader, test_loader, criterion, args.max_epoch)
        trainer.load_weights('./log/%s/teacher/best.pt' % (args.dataset))
        trainer.target_save(tool_loader, './outputs/'+args.dataset+'/train/target/') # only training set can get new target
    elif args.stage == 'student':
        trainer = Trainer(
            optim = optimizer,
            scheduler = exp_lr_scheduler, 
            model = precise_net,
            log_dir = './log/%s/student/' % (args.dataset)
            device = device, 
            rough = False)
        trainer.load_weights('./log/%s/teacher/best.pt' % (args.dataset), epoch=0) # pretrained in the teacher stage
        trainer.train(train_loader, test_loader, criterion, args.max_epoch)

if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epoch', type=int, default=500)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='SHHA')
    parser.add_argument('--stage', type=str, default='rough')
    args = parser.parse_args()
    main(args)

