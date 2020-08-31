#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='cyclegan_in/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='cyclegan_in/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
# netG_A2B.eval()
# netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

images_ds = ImageDataset(opt.dataroot, transforms_=transforms_, mode='train')
dataloader = DataLoader(images_ds,
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('datasets/horse2zebra/distilA2B'):
    os.makedirs('datasets/horse2zebra/distilA2B')
if not os.path.exists('datasets/horse2zebra/distilB2A'):
    os.makedirs('datasets/horse2zebra/distilB2A')
if not os.path.exists('datasets/horse2zebra/distilA2B/A'):
    os.makedirs('datasets/horse2zebra/distilA2B/A')
if not os.path.exists('datasets/horse2zebra/distilA2B/B'):
    os.makedirs('datasets/horse2zebra/distilA2B/B')
if not os.path.exists('datasets/horse2zebra/distilB2A/A'):
    os.makedirs('datasets/horse2zebra/distilB2A/A')
if not os.path.exists('datasets/horse2zebra/distilB2A/B'):
    os.makedirs('datasets/horse2zebra/distilB2A/B')


for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
    real_B = 0.5*(real_B + 1.0)
    real_A = 0.5*(real_A + 1.0)

    # print(fake_B)
    # 1/0

    # Save image files

    if i < len(images_ds.files_A):
        #save_image(real_A, f'datasets/horse2zebra/distilA2B/A/{images_ds.files_A[i].split("/")[-1]}.png')
        save_image(fake_B, f'datasets/horse2zebra/distilA2B/B/{images_ds.files_A[i].split("/")[-1][:-4]}.png')

    if i < len(images_ds.files_B):
        #(fake_A, f'datasets/horse2zebra/distilB2A/A/{images_ds.files_B[i].split("/")[-1]}.png')
        save_image(fake_A, f'datasets/horse2zebra/distilB2A/A/{images_ds.files_B[i].split("/")[-1][:-4]}.png')

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))



sys.stdout.write('\n')
###################################
