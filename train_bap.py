############################################################
#   File: train_bap.py                                     #
#   Created: 2019-11-06 13:22:23                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:train_bap.py                               #
#   Copyright@2019 wvinzh, HUST                            #
############################################################

# system
import os
import time
import shutil
import random
import numpy as np

# my implementation
from model.inception_bap import inception_v3_bap
from model.resnet import resnet50
from dataset.custom_dataset import CustomDataset

from utils import calculate_pooling_center_loss, mask2bbox
from utils import attention_crop, attention_drop, attention_crop_drop
from utils import getDatasetConfig, getConfig
from utils import accuracy, get_lr, save_checkpoint, AverageMeter, set_seed
from utils import Engine

# pytorch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from tensorboardX import SummaryWriter

GLOBAL_SEED = 1231
def _init_fn(worker_id):
    set_seed(GLOBAL_SEED+worker_id)

def train():
    # input params
    set_seed(GLOBAL_SEED)
    config = getConfig()
    data_config = getDatasetConfig(config.dataset)
    sw_log = 'logs/%s' % config.dataset
    sw = SummaryWriter(log_dir=sw_log)
    best_prec1 = 0.
    rate = 0.875

    # define train_dataset and loader
    transform_train = transforms.Compose([
        transforms.Resize((int(config.input_size//rate), int(config.input_size//rate))),
        transforms.RandomCrop((config.input_size,config.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=32./255.,saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = CustomDataset(
        data_config['train'], data_config['train_root'], transform=transform_train)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True, worker_init_fn=_init_fn)

    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = CustomDataset(
        data_config['val'], data_config['val_root'], transform=transform_test)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers, pin_memory=True, worker_init_fn=_init_fn)
    # logging dataset info
    print('Dataset Name:{dataset_name}, Train:[{train_num}], Val:[{val_num}]'.format(
        dataset_name=config.dataset,
        train_num=len(train_dataset),
        val_num=len(val_dataset)))
    print('Batch Size:[{0}], Total:::Train Batches:[{1}],Val Batches:[{2}]'.format(
        config.batch_size, len(train_loader), len(val_loader)
    ))
    # define model
    if config.model_name == 'inception':
        net = inception_v3_bap(pretrained=True, aux_logits=False,num_parts=config.parts)
    elif config.model_name == 'resnet50':
        net = resnet50(pretrained=True,use_bap=True)

    
    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=train_dataset.num_classes)
    net.fc_new = new_linear
    # feature center
    feature_len = 768 if config.model_name == 'inception' else 512
    center_dict = {'center': torch.zeros(
        train_dataset.num_classes, feature_len*config.parts)}

    # gpu config
    use_gpu = torch.cuda.is_available() and config.use_gpu
    if use_gpu:
        net = net.cuda()
        center_dict['center'] = center_dict['center'].cuda()
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and config.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    # define optimizer
    assert config.optim in ['sgd', 'adam'], 'optim name not found!'
    if config.optim == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optim == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # define learning scheduler
    assert config.scheduler in ['plateau',
                                'step'], 'scheduler not supported!!!'
    if config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.1)
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()

    # train val parameters dict
    state = {'model': net, 'train_loader': train_loader,
             'val_loader': val_loader, 'criterion': criterion,
             'center': center_dict['center'], 'config': config,
             'optimizer': optimizer}
    ## train and val
    engine = Engine()
    print(config)
    for e in range(config.epochs):
        if config.scheduler == 'step':
            scheduler.step()
        lr_val = get_lr(optimizer)
        print("Start epoch %d ==========,lr=%f" % (e, lr_val))
        train_prec, train_loss = engine.train(state, e)
        prec1, val_loss = engine.validate(state)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': e + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'center': center_dict['center']
        }, is_best, config.checkpoint_path)
        sw.add_scalars("Accurancy", {'train': train_prec, 'val': prec1}, e)
        sw.add_scalars("Loss", {'train': train_loss, 'val': val_loss}, e)
        if config.scheduler == 'plateau':
            scheduler.step(val_loss)

def test():
    ##
    engine = Engine()
    config = getConfig()
    data_config = getDatasetConfig(config.dataset)
    # define dataset
    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = CustomDataset(
        data_config['val'], data_config['val_root'], transform=transform_test)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers, pin_memory=True)
    # define model
    if config.model_name == 'inception':
        net = inception_v3_bap(pretrained=True, aux_logits=False)
    elif config.model_name == 'resnet50':
        net = resnet50(pretrained=True)

    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=val_dataset.num_classes)
    net.fc_new = new_linear

    # load checkpoint
    use_gpu = torch.cuda.is_available() and config.use_gpu
    if use_gpu:
        net = net.cuda()
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    #checkpoint_path = os.path.join(config.checkpoint_path,'model_best.pth.tar')
    net.load_state_dict(torch.load(config.checkpoint_path)['state_dict'])

    # define loss
    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()
    prec1, prec5 = engine.test(val_loader, net, criterion)


if __name__ == '__main__':
    config = getConfig()
    engine = Engine()
    if config.action == 'train':
        train()
    else:
        test()
