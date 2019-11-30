############################################################
#   File: utils.py                                         #
#   Created: 2019-11-18 20:50:50                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:utils.py                                   #
#   Copyright@2019 wvinzh, HUST                            #
############################################################


import os
import random
import numpy as np
import torch
import shutil
import logging
def getLogger(name='logger',filename=''):
    # 使用一个名字为fib的logger
    logger = logging.getLogger(name)

    # 设置logger的level为DEBUG
    logger.setLevel(logging.DEBUG)

    # 创建一个输出日志到控制台的StreamHandler
    if filename:
        if os.path.exists(filename):
            os.remove(filename)
        hdl = logging.FileHandler(filename, mode='a', encoding='utf-8', delay=False)
    else:
        hdl = logging.StreamHandler()

    format = '%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s'
    datefmt = '%Y-%m-%d(%a)%H:%M:%S'
    formatter = logging.Formatter(format,datefmt)
    hdl.setFormatter(formatter)
    # 给logger添加上handler
    logger.addHandler(hdl)
    return logger
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    set_seed(worker_id)
    # np.random.seed()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
        return old_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, path='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(path, 'model_best.pth.tar'))
        print("Save best model at %s==" %
              os.path.join(path, 'model_best.pth.tar'))