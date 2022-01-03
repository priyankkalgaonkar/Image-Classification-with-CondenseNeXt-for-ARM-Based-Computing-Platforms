#CIFAR-10 Image Classification Script of NXP BlueBox 2.0 using RTMaps.
import rtmaps.types
import numpy as np
from rtmaps.base_component import BaseComponent # base class
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os
import shutil
import time
import math
import warnings
import models
import matplotlib.pyplot as plt
from utils import convert_model, measure_model
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image


# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor

# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")

# Core() is called every time you have a new input
    def Core(self):

        parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
        parser.add_argument('data', metavar='DIR',
                            help='path to dataset')
        parser.add_argument('--model', default='condensenet', type=str, metavar='M',
                            help='model to train the dataset')
        parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=120, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=64, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate (default: 0.1)')
        parser.add_argument('--lr-type', default='cosine', type=str, metavar='T',
                            help='learning rate strategy (default: cosine)',
                            choices=['cosine', 'multistep'])
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--print-freq', '-p', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model (default: false)')
        parser.add_argument('--no-save-model', dest='no_save_model', action='store_true',
                            help='only save best model (default: false)')
        parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                            help='manual seed (default: 0)')
        parser.add_argument('--gpu', default=0,
                            help='gpu available')
        parser.add_argument('--savedir', type=str, metavar='PATH', default='results/savedir',
                            help='path to save result and checkpoint (default: results/savedir)')
        parser.add_argument('--resume', action='store_true',
                            help='use latest checkpoint if have any (default: none)')
        parser.add_argument('--stages', default=4-4-4, type=str, metavar='STAGE DEPTH',
                            help='per layer depth')
        parser.add_argument('--bottleneck', default=4, type=int, metavar='B',
                            help='bottleneck (default: 4)')
        parser.add_argument('--group-1x1', type=int, metavar='G', default=4,
                            help='1x1 group convolution (default: 4)')
        parser.add_argument('--group-3x3', type=int, metavar='G', default=4,
                            help='3x3 group convolution (default: 4)')
        parser.add_argument('--condense-factor', type=int, metavar='C', default=4,
                            help='condense factor (default: 4)')
        parser.add_argument('--growth', default=8-8-8, type=str, metavar='GROWTH RATE',
                            help='per layer growth')
        parser.add_argument('--reduction', default=0.5, type=float, metavar='R',
                            help='transition reduction (default: 0.5)')
        parser.add_argument('--dropout-rate', default=0, type=float,
                            help='drop out (default: 0)')
        parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO',
                            help='group lasso loss weight (default: 0)')
        parser.add_argument('--evaluate', action='store_true',
                            help='evaluate model on validation set (default: false)')
        parser.add_argument('--convert-from', default=None, type=str, metavar='PATH',
                            help='path to saved checkpoint (default: none)')
        parser.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                            help='path to saved checkpoint (default: none)')

        args = parser.parse_args(["--model", "condensenet", "-b", "64", "-j", "12", "cifar10", "--stages", "4-4-4", "--growth", "8-8-8", "--gpu", "0"])

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.stages = list(map(int, args.stages.split('-')))
        args.growth = list(map(int, args.growth.split('-')))
        if args.condense_factor is None:
            args.condense_factor = args.group_1x1

        if args.data == 'cifar1000':
            args.num_classes = 1000
        elif args.data == 'cifar100':
            args.num_classes = 100
        else:
            args.num_classes = 10

        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                                 std=[0.2471, 0.2435, 0.2616])
        train_set = transforms.Compose([
                                         transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         normalize,
                                             ])

        model = models.condensenet(args)
        model = nn.DataParallel(model)
        PATH = "results/path_to_the_trained_weights.pth.tar"
        startTime = time.time()
        model.load_state_dict(torch.load(PATH, map_location=torch.device("cpu"))['state_dict'])

        device = torch.device("cpu")
        model.eval()

        image = Image.open("test_image.jpg")
        #image.show()
        #print(image.filename)
        print(f'Input Image: {image.filename}')

        input = train_set(image)
        input = input.unsqueeze(0)

        model.eval()
        output = model(input)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        topk=(1,5)
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = pred[0].cpu().numpy()[0]
        pred = classes[pred]
        #print(pred)
        print(f'Class Predicted: {pred}')
        executionTime = (time.time() - startTime)
        print('Evaluation Time: ' + str(executionTime) + ' seconds.')

# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass