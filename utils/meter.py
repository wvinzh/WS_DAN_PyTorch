############################################################
#   File: meter.py                                         #
#   Created: 2019-11-05 11:36:34                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:meter.py                                   #
#   Copyright@2019 wvinzh, HUST                            #
############################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count