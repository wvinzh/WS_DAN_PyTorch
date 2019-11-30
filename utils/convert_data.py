############################################################
#   File: convert_data.py                                  #
#   Created: 2019-10-31 19:06:54                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:convert_data.py                            #
#   Copyright@2019 wvinzh, HUST                            #
############################################################

import os
import random
from scipy import io as scio
import argparse
def convert_bird(data_root):
    images_txt = os.path.join(data_root,'images.txt')
    train_val_txt = os.path.join(data_root,'train_test_split.txt')
    labels_txt = os.path.join(data_root,'image_class_labels.txt')

    id_name_dict = {}
    id_class_dict = {}
    id_train_val = {}
    with open(images_txt,'r',encoding='utf-8') as f:
        line = f.readline()
        while line:
            id,name = line.strip().split()
            id_name_dict[id] = name
            line = f.readline()
    
    with open(train_val_txt,'r',encoding='utf-8') as f:
        line = f.readline()
        while line:
            id,trainval = line.strip().split()
            id_train_val[id] = trainval
            line = f.readline()

    with open(labels_txt,'r',encoding='utf-8') as f:
        line = f.readline()
        while line:
            id,class_id = line.strip().split()
            id_class_dict[id] = int(class_id)
            line = f.readline()
    
    train_txt = os.path.join(data_root,'bird_train.txt')
    test_txt = os.path.join(data_root,'bird_test.txt')
    if os.path.exists(train_txt):
        os.remove(train_txt)
    if os.path.exists(test_txt):
        os.remove(test_txt)
    
    f1 = open(train_txt,'a',encoding='utf-8')
    f2 = open(test_txt,'a',encoding='utf-8')

    for id,trainval in id_train_val.items():
        if trainval == '1':
            f1.write('%s %d\n' % (id_name_dict[id],id_class_dict[id]-1))
        else:
            f2.write('%s %d\n' % (id_name_dict[id],id_class_dict[id]-1))
    f1.close()
    f2.close()


def convert_car(data_root):
    train_mat = data_root+'/cars_train_annos.mat'
    test_mat = data_root+'/cars_test_annos_withlabels.mat'
    
    # print(train_data)
    
    train_txt = data_root+'/car_train.txt'
    test_txt = data_root+'/car_test.txt'
    ### train txt
    train_data  = scio.loadmat(train_mat)
    anno = train_data['annotations']
    train_f = open(train_txt,'a')
    for r in anno[0]:
        # print(r,'============')
        _,_,_,_,label,name = r
        # print(label,name)
        train_f.write('%s %d\n' % (name[0],label[0][0]-1))
    train_f.close()

    ### test txt
    test_data  = scio.loadmat(test_mat)
    anno = test_data['annotations']
    test_f = open(test_txt,'a')
    for r in anno[0]:
        # print(r,'============')
        _,_,_,_,label,name = r
        # print(label,name)
        test_f.write('%s %d\n' % (name[0],label[0][0]-1))
    test_f.close()

def convert_aircraft(root):
    # root = '.../Fine-grained/fgvc-aircraft-2013b/data'
    train_txt = root + '/images_variant_trainval.txt'
    test_txt = root + '/images_variant_test.txt'
    variant_txt = root + '/variants.txt'

    ###
    variants_dict = {}
    with open(variant_txt,'r') as f:
        lines = f.readlines()
    index = 0
    for line in lines:
        variant = line.strip()
        if variant in variants_dict:
            continue
        else:
            variants_dict[variant] = index
            index += 1
    # print(index)

    ###
    train_lst = root + '/aircraft_train.txt'
    test_lst = root + '/aircraft_test.txt'

    train_f = open(train_lst,'a')
    with open(train_txt,'r') as f:
        lines = f.readlines()
    for line in lines:
        # print(line)
        lst= line.strip().split(' ',1)
        # print(lst)
        name,label = lst
        name = name +'.jpg'
        label = variants_dict[label]
        train_f.write('%s %d\n'%(name,label))
    train_f.close()
    test_f = open(test_lst,'a')
    with open(test_txt,'r') as f:
        lines = f.readlines()
    for line in lines:
        name,label = line.strip().split(' ',1)
        name = name+'.jpg'
        label = variants_dict[label]
        test_f.write('%s %d\n'%(name,label))
    test_f.close()

def convert_dog(data_root):
    train_lst = data_root+'/train_list.mat'
    train_txt = data_root+'/dog_train.txt'
    info = scio.loadmat(train_lst)['file_list']
    name_dict = {}
    index = 0
    # print(info)
    for i in info:
        # print(i[0])
        name = i[0][0]
        cate = name.split('/')[0]
        if cate in name_dict:
            label = name_dict[cate]
        else:
            label = index
            name_dict[cate] = index
            index += 1
        # print(name,label)
        with open(train_txt,'a') as f:
            f.write('%s %d\n'%(name,label))

    test_lst = data_root+'/test_list.mat'
    test_txt = data_root+'/dog_test.txt'
    info = scio.loadmat(test_lst)['file_list']
    # print(info)
    for i in info:
        # print(i[0])
        name = i[0][0]
        cate = name.split('/')[0]
        label = name_dict[cate]
        # print(name,label)
        with open(test_txt,'a') as f:
            f.write('%s %d\n'%(name,label))


if __name__ == '__main__':
    # convert_bird('/home/XXX/Dataset/Fine-grained/CUB_200_2011')
    # convert_car('/home/XXX/Dataset/Fine-grained/Car/devkit')
    # convert_aircraft('/home/XXX/Dataset/Fine-grained/fgvc-aircraft-2013b/data')
    # convert_dog('/home/XXX/Dataset/Fine-grained/dogs')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',type=str,default='bird')
    parser.add_argument('--root_path',type=str,default='.')
    arg = parser.parse_args()
    func = eval('convert_'+arg.dataset_name)
    func(arg.root_path)
    