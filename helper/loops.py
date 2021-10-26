from __future__ import print_function, division

import sys
import time
import torch

import torch.nn as nn

from .util import AverageMeter, accuracy
from helper.util import cluster




def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    
    # set modules as train()
    for module in module_list:
        module.train()
    
    # set teacher as eval()
    module_list[-1].eval()

    #criterion_cls = criterion_list[0]
    criterion_kl = criterion_list[1]
    criterion_mse = criterion_list[2]
    criterion_sp = criterion_list[3]
    
    softmax = nn.Softmax(dim=1).cuda()


    model_s = module_list[0]
    model_t = module_list[-1]
    
    try:
        context_new = torch.zeros( model_s.fc.weight.shape, dtype=torch.float32).cuda()   
        current_num = torch.zeros(model_s.fc.weight.shape[0], dtype=torch.float32)
        class_num = model_s.fc.weight.shape[0]
    except:
        try:
            context_new = torch.zeros( model_s.linear.weight.shape, dtype=torch.float32).cuda()   
            current_num = torch.zeros(model_s.linear.weight.shape[0], dtype=torch.float32)  
            class_num = model_s.linear.weight.shape[0]
        except:
            context_new = torch.zeros( model_s.classifier.weight.shape, dtype=torch.float32).cuda()   
            current_num = torch.zeros(model_s.classifier.weight.shape[0], dtype=torch.float32)  
            class_num = model_s.classifier.weight.shape[0]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    for idx, data in enumerate(train_loader):
  
        input, target, index = data
        
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
  
        # ===================forward=====================
        preact = False

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]
            


        if epoch==opt.init_epochs:   
            
            fea_s  = feat_s[opt.hint_layer].detach()
            
            soft_t = softmax(logit_t/opt.net_T)
            
            for i in range( len(target) ):
                context_new[target[i]] = context_new[target[i]]*( current_num[target[i]]/(current_num[target[i]]+soft_t[i][target[i]]) )+ fea_s[i]*(soft_t[i][target[i]]/(current_num[target[i]]+soft_t[i][target[i]]) )
                current_num[target[i]]+= soft_t[i][target[i]]
        
        # loss
        loss_kl = criterion_kl(logit_s, logit_t)
        
        fea_reg = module_list[2]
        f_s = fea_reg(feat_s[opt.hint_layer])
        f_t = feat_t[opt.hint_layer]
            
        loss_sample = criterion_mse(f_s, f_t)
        
        list_s, list_t = cluster(feat_s[opt.hint_layer], f_t, target, class_num)
            
        involve_class = 0
            
        loss_class=0.0
            
        for k in range( len(list_s) ):
            
            cur_len = len( list_s[k] )
            
            if cur_len>=2:
                cur_f_s = torch.stack(list_s[k])
                cur_f_t = torch.stack(list_t[k])
                    
                loss_class+= criterion_sp(cur_f_s, cur_f_t) 
                    
                involve_class += 1
                    
                    
        if involve_class==0:
            loss_class = 0.0
        else:
            loss_class = loss_class/involve_class   
        

        loss =  opt.aa*loss_kl + opt.bb * loss_sample + opt.cc * loss_class 

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    return top1.avg, losses.avg, context_new



def train_distill_context(epoch, train_loader, module_list, criterion_list, optimizer, opt, context):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    
    # set teacher as eval()
    module_list[-1].eval()


    context_new = torch.zeros(context.shape, dtype=torch.float32).cuda()
   
    current_num = torch.zeros(context.shape[0], dtype=torch.float32)


    criterion_cls = criterion_list[0]
    criterion_kl = criterion_list[1]
    criterion_mse = criterion_list[2]
    criterion_sp = criterion_list[3]
    
    softmax = nn.Softmax(dim=1).cuda()
    

    model_s = module_list[0]
    model_s_fc_new = module_list[1]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    for idx, data in enumerate(train_loader):
  
        input, target, index = data
        
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
  
        # ===================forward=====================
        preact = False

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        
        
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]
        
        fea_s  = feat_s[opt.hint_layer].detach()
        
        soft_t = softmax(logit_t/opt.net_T)
        
        for i in range( len(target) ):
            context_new[target[i]] = context_new[target[i]]*( current_num[target[i]]/(current_num[target[i]]+soft_t[i][target[i]]) ) + fea_s[i]*(soft_t[i][target[i]]/(current_num[target[i]]+soft_t[i][target[i]]) )
            current_num[target[i]]+=soft_t[i][target[i]]
        
        
        p = softmax(logit_s.detach()/opt.net_T)
        
        sam_contxt = torch.mm(p, context)
        
        f_new = torch.cat((feat_s[opt.hint_layer], sam_contxt),1)
        
        logit_s_new = model_s_fc_new(f_new)
        
        
        loss_cls = criterion_cls(logit_s_new, target)
        
        loss_kl = criterion_kl(logit_s, logit_t)
        
        
        class_num = model_s_fc_new.linear.weight.shape[0]
        
        fea_reg = module_list[2]
        f_s = fea_reg(feat_s[opt.hint_layer])
        f_t = feat_t[opt.hint_layer]
            
        loss_sample = criterion_mse(f_s, f_t)
            
        list_s, list_t = cluster(feat_s[opt.hint_layer], f_t, target, class_num)
            
        involve_class = 0
            
        loss_class=0.0
            
        for k in range( len(list_s) ):
            
            cur_len = len( list_s[k] )
            
            if cur_len>=2:
                
                cur_f_s = torch.stack(list_s[k])
                cur_f_t = torch.stack(list_t[k])
                    
                loss_class+= criterion_sp(cur_f_s, cur_f_t) 
                    
                involve_class += 1
                    
                    
        if involve_class==0:
            loss_class = 0.0
        else:
            loss_class = loss_class/involve_class  


        loss =  opt.aa*(loss_cls + loss_kl) + opt.bb * loss_sample + opt.cc * loss_class

        acc1, acc5 = accuracy(logit_s_new, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    
    context_new = opt.cu*context + (1-opt.cu)*context_new

    return top1.avg, losses.avg, context_new


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


        print(' * test Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate_st(val_loader, model, criterion, opt, context, model_fc_new):
    """validation"""
    batch_time = AverageMeter()
    
    top1_new = AverageMeter()
    top5_new = AverageMeter()
    
    softmax = nn.Softmax(dim=1).cuda()

    # switch to evaluate mode
    model.eval()
    model_fc_new.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                

            # compute output
            feat, output  = model(input, is_feat=True, preact=False)
            
            p = softmax(output/opt.net_T)
        
            sam_contxt = torch.mm(p, context)
        
            f_new = torch.cat((feat[opt.hint_layer], sam_contxt),1)
        
            output_new = model_fc_new(f_new)            
            

            acc1_new, acc5_new = accuracy(output_new, target, topk=(1, 5))
            top1_new.update(acc1_new[0], input.size(0))
            top5_new.update(acc5_new[0], input.size(0))           
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Test Acc@1 {top1_new.avg:.3f} Acc@5 {top5_new.avg:.3f}'
              .format(top1_new=top1_new, top5_new=top5_new))
        
    return top1_new.avg, top5_new.avg
