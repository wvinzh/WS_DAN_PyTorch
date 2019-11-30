# PyTorch Implementation Of WS-DAN

## Introduction
This is a PyTorch implementation of the paper
"[See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification](https://arxiv.org/abs/1901.09891)". It also has an official TensorFlow implementation [WS_DAN](https://github.com/tau-yihouxiang/WS_DAN). The core part of the code refers to the official version, and finally，the  performance almost reaches the results reported in the paper.

## Environment

- Ubuntu 16.04, GTX 1080 8G * 2, cuda 8.0
- Anaconda with Python=3.6.5, PyTorch=0.4.1, torchvison=0.2.1, etc.
- Some **third-party dependencies** may be installed with **pip** or **conda** when needed.

## Result

| Dataset       | ACC(this repo)    | ACC Refine(this repo) | ACC(paper)
| ------------- | ------ | ----------- | ----------- |
| CUB-200-2011  | 88.20 | 89.30      | 89.4
| FGVC-Aircraft | 93.15 | 93.22      | 93.0
| Stanford Cars | 94.13 |  94.43   | 94.5
| Stanford Dogs | 86.03 | 86.46     | 92.2

You can download pretrained models from [WS_DAN_Onedrive](https://1drv.ms/f/s!AseTbxZ7P87UknnvrfLAsIFlhAmb)

## Install

1. Clone the repo
```
git clone https://github.com/wvinzh/WS_DAN_PyTorch
```
2. Prepare dataset

- Download the following datasets. 

Dataset | Object | Category | Training | Testing
---|--- |--- |--- |---
[CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) | Bird | 200 | 5994 | 5794
[Stanford-Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | Car | 100 | 6667 | 3333 
[fgvc-aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) | Aircraft | 196 | 8144 | 8041
[Stanford-Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) | Dogs | 120 | 12000 | 8580

- Extract the data like following:
```
Fine-grained
├── CUB_200_2011
│   ├── attributes
│   ├── bounding_boxes.txt
│   ├── classes.txt
│   ├── image_class_labels.txt
│   ├── images
│   ├── images.txt
│   ├── parts
│   ├── README
├── Car
│   ├── cars_test
│   ├── cars_train
│   ├── devkit
│   └── tfrecords
├── fgvc-aircraft-2013b
│   ├── data
│   ├── evaluation.m
│   ├── example_evaluation.m
│   ├── README.html
│   ├── README.md
│   ├── vl_argparse.m
│   ├── vl_pr.m
│   ├── vl_roc.m
│   └── vl_tpfp.m
├── dogs
│   ├── file_list.mat
│   ├── Images
│   ├── test_list.mat
│   └── train_list.mat
```
- Prepare the ./data folder: generate file list txt (**using ./utils/convert_data.py**) and do soft link. 
```
python utils/convert_data.py  --dataset_name bird --root_path .../Fine-grained/CUB_200_2011
```
    
```
├── data
│   ├── Aircraft -> /your_root_path/Fine-grained/fgvc-aircraft-2013b/data
│   ├── aircraft_test.txt
│   ├── aircraft_train.txt
│   ├── Bird -> /your_root_path/Fine-grained/CUB_200_2011
│   ├── bird_test.txt
│   ├── bird_train.txt
│   ├── Car -> /your_root_path/Fine-grained/Car
│   ├── car_test.txt
│   ├── car_train.txt
│   ├── Dog -> /your_root_path/Fine-grained/dogs
│   ├── dog_test.txt
│   └── dog_train.txt

```



## Usage

- Train

```
python train_bap.py train\
    --model-name inception \
    --batch-size 12 \
    --dataset car \
    --image-size 512 \
    --input-size 448 \
    --checkpoint-path checkpoint/car \
    --optim sgd \
    --scheduler step \
    --lr 0.001 \
    --momentum 0.9 \
    --weight-decay 1e-5 \
    --workers 4 \
    --parts 32 \
    --epochs 80 \
    --use-gpu \
    --multi-gpu \
    --gpu-ids 0,1 \
```
A simple way is to use `sh train_bap.sh` or run backgroud with logs using cmd `nohup sh train_bap.sh 1>train.log 2>error.log &`
- Test

```
python train_bap.py test\
    --model-name inception \
    --batch-size 12 \
    --dataset car \
    --image-size 512 \
    --input-size 448 \
    --checkpoint-path checkpoint/car/model_best.pth.tar \
    --workers 4 \
    --parts 32 \
    --use-gpu \
    --multi-gpu \
    --gpu-ids 0,1 \
```

