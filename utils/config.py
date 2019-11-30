import os
import argparse


def getConfig():
    parser = argparse.ArgumentParser()

    # train or test
    # action = parser.add_subparsers()
    # action.add_parser('train', action='store_true', help='run train')
    # action.add_parser('test', action='store_true', help='run test')
    parser.add_argument('action', choices=('train', 'test'))
    # dataset
    parser.add_argument('--dataset', metavar='DIR',
                        default='bird', help='name of the dataset')
    parser.add_argument('--image-size', '-i', default=512, type=int,
                        metavar='N', help='image size (default: 512)')
    parser.add_argument('--input-size', '-cs', default=448, type=int,
                        metavar='N', help='the input size of the model (default: 448)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # optimizer config
    parser.add_argument('--optim', default='sgd', type=str,
                        help='the name of optimizer(adam,sgd)')
    parser.add_argument('--scheduler', default='plateau', type=str,
                        help='the name of scheduler(step,plateau)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')

    # model config
    parser.add_argument('--parts', default=32, type=int,
                        metavar='N', help='number of parts (default: 32)')
    parser.add_argument('--alpha', default=0.95, type=float,
                        metavar='N', help='weight for BAP loss')
    parser.add_argument('--model-name', default='inception', type=str,
                        help='model name')

    # training config
    parser.add_argument('--use-gpu', action="store_true", default=True,
                        help='whether use gpu or not, default True')
    parser.add_argument('--multi-gpu', action="store_true", default=True,
                        help='whether use multiple gpus or not, default True')
    parser.add_argument('--gpu-ids', default='0,1',
                        help='gpu id list(eg: 0,1,2...)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--print-freq', '-pf', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint-path', default='checkpoint', type=str, metavar='checkpoint_path',
                        help='path to save checkpoint (default: checkpoint)')

    args = parser.parse_args()

    return args


def getDatasetConfig(dataset_name):
    assert dataset_name in ['bird', 'car',
                            'aircraft','dog'], 'No dataset named %s!' % dataset_name
    dataset_dict = {
        'bird': {'train_root': 'data/Bird/images',  # the root path of the train images stored
                 'val_root': 'data/Bird/images',  # the root path of the validate images stored
                                                    # training list file (aranged as filename lable)
                 'train': 'data/bird_train.txt',
                 'val': 'data/bird_test.txt'},  # validate list file
        'car': {'train_root': 'data/Car/cars_train',
                'val_root': 'data/Car/cars_test',
                'train': 'data/car_train.txt',
                'val': 'data/car_test.txt'},
        'aircraft': {'train_root': 'data/Aircraft/images',
                     'val_root': 'data/Aircraft/images',
                     'train': 'data/aircraft_train.txt',
                     'val': 'data/aircraft_test.txt'},
        'dog': {'train_root': 'data/Dog/Images',
                'val_root': 'data/Dog/Images',
                'train': 'data/dog_train.txt',
                'val': 'data/dog_test.txt'},
    }
    return dataset_dict[dataset_name]


if __name__ == '__main__':
    config = getConfig()
    config = vars(config)
    dataConfig = getDatasetConfig('bird')
    # for k,v in config.items():
    #     print(k,v)
    # config.
    print(config)
