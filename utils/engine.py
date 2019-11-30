############################################################
#   File: engine.py                                        #
#   Created: 2019-11-20 15:02:13                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:engine.py                                  #
#   Copyright@2019 wvinzh, HUST                            #
############################################################
import time
from utils import calculate_pooling_center_loss, mask2bbox
from utils import attention_crop, attention_drop, attention_crop_drop
from utils import getDatasetConfig, getConfig, getLogger
from utils import accuracy, get_lr, save_checkpoint, AverageMeter, set_seed

import torch
import torch.nn.functional as F

class Engine():
    def __init__(self,):
        pass

    def train(self,state,epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        config = state['config']
        print_freq = config.print_freq
        model = state['model']
        criterion = state['criterion']
        optimizer = state['optimizer']
        train_loader = state['train_loader']
        model.train()
        end = time.time()
        for i, (img, label) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = label.cuda()
            input = img.cuda()
            # compute output
            attention_maps, raw_features, output1 = model(input)
            features = raw_features.reshape(raw_features.shape[0], -1)

            feature_center_loss, center_diff = calculate_pooling_center_loss(
                features, state['center'], target, alfa=config.alpha)

            # update model.centers
            state['center'][target] += center_diff

            # compute refined loss
            # img_drop = attention_drop(attention_maps,input)
            # img_crop = attention_crop(attention_maps, input)
            img_crop, img_drop = attention_crop_drop(attention_maps, input)
            _, _, output2 = model(img_drop)
            _, _, output3 = model(img_crop)

            loss1 = criterion(output1, target)
            loss2 = criterion(output2, target)
            loss3 = criterion(output3, target)

            loss = (loss1+loss2+loss3)/3 + feature_center_loss
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output1, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
                print("loss1,loss2,loss3,feature_center_loss", loss1.item(), loss2.item(), loss3.item(),
                    feature_center_loss.item())
        return top1.avg, losses.avg
    
    def validate(self,state):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        config = state['config']
        print_freq = config.print_freq
        model = state['model']
        val_loader = state['val_loader']
        criterion = state['criterion']
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda()
                input = input.cuda()
                # forward
                attention_maps, raw_features, output1 = model(input)
                features = raw_features.reshape(raw_features.shape[0], -1)
                feature_center_loss, _ = calculate_pooling_center_loss(
                    features, state['center'], target, alfa=config.alpha)

                img_crop, img_drop = attention_crop_drop(attention_maps, input)
                # img_drop = attention_drop(attention_maps,input)
                # img_crop = attention_crop(attention_maps,input)
                _, _, output2 = model(img_drop)
                _, _, output3 = model(img_crop)
                loss1 = criterion(output1, target)
                loss2 = criterion(output2, target)
                loss3 = criterion(output3, target)
                # loss = loss1 + feature_center_loss
                loss = (loss1+loss2+loss3)/3+feature_center_loss
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output1, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))

            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

        return top1.avg, losses.avg

    def test(self,val_loader, model, criterion):
        top1 = AverageMeter()
        top5 = AverageMeter()
        print_freq = 100
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda()
                input = input.cuda()
                # forward
                attention_maps, _, output1 = model(input)
                refined_input = mask2bbox(attention_maps, input)
                _, _, output2 = model(refined_input)
                output = (F.softmax(output1, dim=-1)+F.softmax(output2, dim=-1))/2
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader),
                            top1=top1, top5=top5))

            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

        return top1.avg, top5.avg


if __name__ == '__main__':

    engine = Engine()
    engine.train()
