"""
training framework of CID
"""

from __future__ import print_function

import os
import argparse
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Reg

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate

from distiller_zoo import KL, Cosine
from distiller_zoo import NORM_MSE

from helper.loops import train_distill as train_init, validate, validate_st, train_distill_context as train
from helper.util import set_seed


def parse_option():

    parser = argparse.ArgumentParser('Arguments for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=240, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=20, help='init training for methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_10_10','wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'MobileNetV2', 'ShuffleV1',
                                 'ShuffleV2', 'ResNet34', 'wrn_16_4', 'wrn_40_4', 'wrn_16_10', 'ResNet10'])
    
    parser.add_argument('--path_t', type=str, default=None, help='teacher model')
    
    # distillation
    parser.add_argument('--distill', type=str, default='CID', choices=['CID'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-a', '--aa', type=float, default=1, help='weight for classification')
    parser.add_argument('-b', '--bb', type=float, default=None, help='weight balance for sample')
    parser.add_argument('-c', '--cc', type=float, default=None, help='weight balance for class')
        
    parser.add_argument('-NT', '--net_T', type=float, default=4, help='net Tempereture')
    
    parser.add_argument('-s', '--seed', type=int, default=1, help='seed')
    
    parser.add_argument('-u', '--cu', type=float, default=0, help='moving average cofficient')

    # last layer
    parser.add_argument('--hint_layer', default=-1, type=int, choices=[-1])

    opt = parser.parse_args()

    # set different learning rate fro these models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01
    

    # set the path according to the environment
    opt.model_path = './save/student_model'
  
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_{}_{}_a:{}_b:{}_c:_{}{}'.format(opt.model_s, opt.dataset, opt.distill,
                        opt.aa, opt.bb, opt.cc, opt.trial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    
    
    best_acc = 0

    opt = parse_option()
    
    print(opt)
    
    set_seed(opt.seed)


    # dataloader
    if opt.dataset == 'cifar100':
  
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _= model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    
    trainable_list = nn.ModuleList([])
    trainable_list2 = nn.ModuleList([])
    
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = KL(opt.net_T)

    if opt.distill == 'CID':
        
        criterion_kd = NORM_MSE()
        
        criterion_cc = Cosine()         

        _, Cs_h = feat_s[opt.hint_layer].shape
        
        model_s_fc_new = Reg( Cs_h*2, n_cls)
        
        module_list.append(model_s_fc_new)
        
        trainable_list.append(model_s_fc_new)
        
        _, Ct_h = feat_t[opt.hint_layer].shape
        
        Reger_fea = Reg( Cs_h, Ct_h)
        
        module_list.append(Reger_fea)
        
        trainable_list2.append(Reger_fea)
        
    else:
        raise NotImplementedError(opt.distill)

        

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)

    criterion_list.append(criterion_kd)
    criterion_list.append(criterion_cc)


    # optimizer
    optimizer = optim.SGD([{'params': trainable_list.parameters()}, {'params': trainable_list2.parameters(), 'weight_decay': 0.0}],
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,
                          nesterov=True)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)


    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        
        if epoch <= opt.init_epochs:
            train_acc, train_loss, context = train_init(epoch, train_loader, module_list, criterion_list, optimizer, opt)
            context_old = context
        else:
            context_old = context
            train_acc, train_loss, context = train(epoch, train_loader, module_list, criterion_list, optimizer, opt, context)
        
        if train_loss!=train_loss:
            return
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

      
        test_acc, tect_acc_top5 = validate_st(val_loader, model_s, criterion_cls, opt, context_old, model_s_fc_new)


        
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'model_s_fc_new': model_s_fc_new.state_dict(),
                'context_old': context_old,
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
                'model_s_fc_new': model_s_fc_new.state_dict(),
                'context_old': context_old
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    print('best accuracy:', best_acc.cpu().numpy())

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'model_s_fc_new': model_s_fc_new.state_dict(),
        'context_old': context_old
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
