############################################################
#   File: custom_dataset.py                                #
#   Created: 2019-10-31 19:28:59                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:custom_dataset.py                          #
#   Copyright@2019 wvinzh, HUST                            #
############################################################

from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision
import os
import random

class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, training=False):
        self.image_list = []
        self.id_list = []
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 0
        self.training = training
        with open(txt_file, 'r') as f:
            line = f.readline()
            # self.datas = f.readlines()
            while line:
                img_name = line.split()[0]
                label = int(line.split()[1])
                # label = int(label)
                self.image_list.append(img_name)
                self.id_list.append(label)
                line = f.readline()
        self.num_classes = max(self.id_list)+1
        
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = self.id_list[idx]
        img_name = os.path.join(self.root_dir,img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image,label


def test_dataset():
    root = '/home/zengh/Dataset/Fine-grained/CUB_200_2011/images'
    txt = '/home/zengh/Dataset/Fine-grained/CUB_200_2011/test_pytorch.txt'
    from torchvision import transforms
    rgb_mean = [0.5,0.5,0.5]
    rgb_std = [0.5,0.5,0.5]
    transform_val = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])
    carData = CustomDataset(txt,root,transform_val,True)
    print(carData.num_classes)
    dataloader = DataLoader(carData,batch_size=16,shuffle=True)
    for data in dataloader:
        images,labels = data
        # print(images.size(),labels.size(),labels)


if __name__=='__main__':
    test_dataset()