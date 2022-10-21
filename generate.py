from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json

import models.dcgan as dcgan
import models.mlp as mlp

if __name__=="__main__":
    #构建参数对象与解析
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='path to generator config .json file')
    parser.add_argument('-w', '--weights', required=True, type=str, help='path to generator weights .pth file')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help="path to to output directory")
    parser.add_argument('-n', '--nimages', required=True, type=int, help="number of images to generate", default=1)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    opt = parser.parse_args()

    # 加载生成器的配置文件generator_config.json
    with open(opt.config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())
    # 读取配置文件
    imageSize = generator_config["imageSize"]
    nz = generator_config["nz"]
    nc = generator_config["nc"]
    ngf = generator_config["ngf"]
    noBN = generator_config["noBN"]
    ngpu = generator_config["ngpu"]
    mlp_G = generator_config["mlp_G"]
    n_extra_layers = generator_config["n_extra_layers"]

    # 选择训练方法
    if noBN:
        netG = dcgan.DCGAN_G_nobn(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif mlp_G:
        netG = mlp.MLP_G(imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # 加载训练出来的权重参数文件，比如：netG_epoch_3.pth
    netG.load_state_dict(torch.load(opt.weights))

    # 初始化噪音数据
    fixed_noise = torch.FloatTensor(opt.nimages, nz, 1, 1).normal_(0, 1)

    #是否启用CUDA
    if opt.cuda:
        netG.cuda()
        fixed_noise = fixed_noise.cuda()

    fake = netG(fixed_noise)
    fake.data = fake.data.mul(0.5).add(0.5)#乘以0.5加上0.5
    
    # 转换形状并生成指定数量的图片
    for i in range(opt.nimages):
        vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(opt.output_dir, "generated_%02d.png"%i))
