"""
Evaluation
"""

from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import model_dict
from models.util import Reg
from dataset.cifar100 import get_cifar100_dataloaders
from helper.loops import validate_st
import os


def parse_option():

    parser = argparse.ArgumentParser('Arguments for training')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('-NT', '--net_T', type=float, default=4, help='net Tempereture')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_10_10','wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'MobileNetV2', 'ShuffleV1',
                                 'ShuffleV2', 'ResNet34', 'wrn_16_4', 'wrn_40_4', 'wrn_16_10', 'ResNet10'])
    
    parser.add_argument('--model_path', type=str, default=None, help='student model')
    
    parser.add_argument('--hint_layer', default=-1, type=int, choices=[-1])
    
    opt = parser.parse_args()

    return opt


def main():
    
    opt = parse_option()
    
    print(opt)

    # dataloader
    if opt.dataset == 'cifar100':
  
        _, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                         num_workers=opt.num_workers,
                                                         is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    model_s.load_state_dict(torch.load(opt.model_path)['model'])

    data = torch.randn(2, 3, 32, 32)
    
    model_s.eval()

    feat_s, _= model_s(data, is_feat=True)
    
    _, Cs_h = feat_s[opt.hint_layer].shape
        
    model_s_fc_new = Reg( Cs_h*2, n_cls)
    model_s_fc_new.load_state_dict(torch.load(opt.model_path)['model_s_fc_new'])
    
    context = torch.load(opt.model_path)['context_old']
    
    criterion_cls = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model_s_fc_new.cuda()
        model_s.cuda()
        context = context.cuda()
        criterion_cls.cuda()
        cudnn.benchmark = True
        
        
    test_acc, tect_acc_top5 = validate_st(val_loader, model_s, criterion_cls, opt, context, model_s_fc_new)

if __name__ == '__main__':
    main()
