#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--use_bn', action='store_true', help='Use of Batchnorm / Instancenorm')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--output_dir', type=str, default='output', help='where to save the results')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--mode', default="A2B", help='mode of distillation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--eval', type=int, default=200, help='epoch to go to eval mode')
parser.add_argument('--n_blocks', type=int, default=9, help='number of residual blocks')
opt = parser.parse_args()
print(opt)

import os
if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
if opt.mode == "A2B":
    Gen = Generator(opt.input_nc, opt.output_nc, use_bn=opt.use_bn, n_residual_blocks=opt.n_blocks)
elif opt.mode == "B2A":
    Gen = Generator(opt.output_nc, opt.input_nc, use_bn=opt.use_bn, n_residual_blocks=opt.n_blocks)
else:
    raise

device = "cpu"
if opt.cuda:
    device = "cuda"
    Gen.cuda()

Gen.apply(weights_init_normal)

# Lossess
criterion = torch.nn.MSELoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(Gen.parameters(),
                                lr=opt.lr, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode="distil" + opt.mode, unaligned=False), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    if (opt.eval == epoch):
        Gen.eval()
    for i, batch in enumerate(dataloader):
        optimizer_G.zero_grad()

        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        if (opt.mode=="A2B"):
            gen_img = Gen(real_A)
            loss = criterion(gen_img, real_B)
        else:
            gen_img = Gen(real_B)
            loss = criterion(gen_img, real_A)

        loss.backward()
        optimizer_G.step()
        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss}, images={'real_A': real_A, 'real_B': real_B, 'gen': gen_img})

    # Update learning rates
    lr_scheduler_G.step()

    # Save models checkpoints
    torch.save(Gen.state_dict(), f'{opt.output_dir}/Gen.pth')
###################################
