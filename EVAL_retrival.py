import os
import numpy as np 
import torch
import torch.nn as nn
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime as dt
import json
import csv

# from JSCC_get_args import get_args_parser
# from JSCC_modules import *
# from dataset import CIFAR10, ImageNet, Kodak
# from JSCC_utils import *
# from JSCC_coop_network import *
# from DHD_models import *
# from DHD_loss import *
# from DHD_Retrieval import DoRetrieval
# from DHD_modifiedALEX import ModifiedAlexNet
# from Dataloader import Loader
from source_DHD import *
from source_JSCC_MIMO import *

import matplotlib.pyplot as plt
import datetime

from PIL import Image
# import imagehash

#########
#           Parameter Setting
#########

dname = 'nuswide'
path = './datasets/'
args = get_args_parser()

Img_dir = path + dname + '256'
Train_dir = path + dname + '_Train.txt'
Gallery_dir = path + dname + '_DB.txt'
Query_dir = path + dname + '_Query.txt'
org_size = 256
input_size = 224
NB_CLS = 21

# Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS)
# Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)
Query_set = Loader(Img_dir, Query_dir, NB_CLS)
# Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)
Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=64, num_workers=args.DHD_num_workers)
valid_loader = Query_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

###########
#           Load DHD Model
###########

def load_checkpoint(model, filename):
    """Load checkpoint"""
    checkpoint = T.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint['epoch'], checkpoint['best_mAP']

def load_nets(path, nets):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        nets.load_state_dict(checkpoint['jscc_model'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        print(f"No checkpoint found at {path}")
        epoch = 0
    
    return epoch

class Hash_func(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func, self).__init__()
        self.Hash = nn.Sequential(
            nn.Linear(fc_dim, N_bits, bias=False),
            nn.LayerNorm(N_bits))
        self.P = nn.Parameter(torch.FloatTensor(NB_CLS, N_bits), requires_grad=True)
        nn.init.xavier_uniform_(self.P, gain=nn.init.calculate_gain('tanh'))

    def forward(self, X):
        X = self.Hash(X)
        return torch.tanh(X)

def get_discriminator(args,discriminator_path):
    if args.DHD_encoder == 'AlexNet':
        Baseline = AlexNet()
        fc_dim = 4096
    else:
        print("Wrong encoder name.")
        return None

    H = Hash_func(fc_dim, args.DHD_N_bits, NB_CLS=21)
    net = nn.Sequential(Baseline, H)
    net = nn.DataParallel(net)  # Add this line to wrap model for DataParallel
    net.to(device)

    # checkpoint_path = 'checkpoint.pth.tar'
    epoch, best_mAP = load_checkpoint(net, discriminator_path)
    print("best_mAP of" + discriminator_path+": "+ str(best_mAP))

    return net, H

###########
#           Load JSCC Model
###########

def get_generator(jscc_args, job_name,checkpoint_path=None):
    if jscc_args.JSCC_diversity:
        enc = EncoderCell(c_in=3, c_feat=jscc_args.JSCC_cfeat, c_out=jscc_args.JSCC_cout, attn=jscc_args.JSCC_adapt).to(jscc_args.device)
        dec = DecoderCell(c_in=jscc_args.JSCC_cout, c_feat=jscc_args.JSCC_cfeat, c_out=3, attn=jscc_args.JSCC_adapt).to(jscc_args.device)
        jscc_model = Div_model(jscc_args, enc, dec)
    else:
        enc = EncoderCell(c_in=3, c_feat=jscc_args.JSCC_cfeat, c_out=2*jscc_args.JSCC_cout, attn=jscc_args.JSCC_adapt).to(jscc_args.JSCC_device)
        dec = DecoderCell(c_in=2*jscc_args.JSCC_cout, c_feat=jscc_args.JSCC_cfeat, c_out=3, attn=jscc_args.JSCC_adapt).to(jscc_args.JSCC_device)
        if jscc_args.JSCC_res:
            res = EQUcell(6*jscc_args.JSCC_Nr, 128, 4).to(jscc_args.device)
            jscc_model = Mul_model(jscc_args, enc, dec, res)
        else:
            jscc_model = Mul_model(jscc_args, enc, dec)
    
    if jscc_args.JSCC_resume:
        load_weights(job_name, jscc_model)

    jscc_model = nn.DataParallel(jscc_model)  # Add this line to wrap model for DataParallel
    if checkpoint_path != None:
        # checkpoint_path = './models/train_JSCC_model_with_nuswide.pth'
        epoch = load_nets(checkpoint_path, jscc_model)
    
    return jscc_model

###########
#           Main Function
###########

def save_tensor_as_jpg(tensor, output_dir, base_filename):
    """
    保存张量为JPEG图片。

    参数:
    tensor (torch.Tensor): 形状为 (B, C, H, W) 的图像张量
    output_dir (str): 保存图片的目录
    base_filename (str): 保存图片的基本文件名（不包括扩展名）

    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 转换为 PIL 图像
    to_pil = transforms.ToPILImage()

    for i in range(tensor.size(0)):
        image_pil = to_pil(tensor[i].cpu())
        image_pil.save(os.path.join(output_dir, f"{base_filename}_{i}.jpg"))

def tensor_to_pil(tensor):
    # Convert a tensor to a PIL Image
    return transforms.ToPILImage()(tensor.cpu())

def evaluate_gan(discriminator_args, generator_args, job_name, generator_load_path, discriminator_path,SNR,for_image = False):
    # os.environ["CUDA_VISIBLE_DEVICES"] = discriminator_args.DHD_gpu_id
    discriminator, H = get_discriminator(discriminator_args, discriminator_path)

    discriminator.eval()

    num_classes = 21

    AugT = Augmentation(256, discriminator_args.DHD_transformation_scale)  # Assuming org_size is 32 for CIFAR-10
    Crop = nn.Sequential(Kg.CenterCrop(224))
    Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

    n_critic = 1
    max_epoch = discriminator_args.DHD_max_epoch

    # evaluation metrics for JSON file
    evaluation_metrics = []

    # time stamp
    timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
    json_filename = f'LOG/FINAL_EVAL/{job_name}.json'

    HD_criterion = HashDistill()

    # Initial epoch
    epoch = 0

    C_loss = 0.0
    gallery_codes, gallery_labels = [], []

    hash_distance_list = []
    hamming_dist_list = []

    mAP = DoRetrieval(device, discriminator.eval(), "./datasets/nuswide256", "./datasets/nuswide_DB.txt", "./datasets/nuswide_Query.txt", num_classes, 5000, discriminator_args)
    mAP_value = mAP.item()  # convert tensor to a Python number
    print("TOP 5000 retrieval mAP: ", mAP_value)

    mAP = DoRetrieval(device, discriminator.eval(), "./datasets/nuswide256", "./datasets/nuswide_DB.txt", "./datasets/nuswide_Query.txt", num_classes, 4000, discriminator_args)
    mAP_value = mAP.item()  # convert tensor to a Python number
    print("TOP 4000 retrieval mAP: ", mAP_value)

    mAP = DoRetrieval(device, discriminator.eval(), "./datasets/nuswide256", "./datasets/nuswide_DB.txt", "./datasets/nuswide_Query.txt", num_classes, 3000, discriminator_args)
    mAP_value = mAP.item()  # convert tensor to a Python number
    print("TOP 3000 retrieval mAP: ", mAP_value)  

    mAP = DoRetrieval(device, discriminator.eval(), "./datasets/nuswide256", "./datasets/nuswide_DB.txt", "./datasets/nuswide_Query.txt", num_classes, 2000, discriminator_args)
    mAP_value = mAP.item()  # convert tensor to a Python number
    print("TOP 2000 retrieval mAP: ", mAP_value)  

    mAP = DoRetrieval(device, discriminator.eval(), "./datasets/nuswide256", "./datasets/nuswide_DB.txt", "./datasets/nuswide_Query.txt", num_classes, 1000, discriminator_args)
    mAP_value = mAP.item()  # convert tensor to a Python number
    print("TOP 1000 retrieval mAP: ", mAP_value)  





if __name__ == '__main__':
    args1 = args
    # SNR = [-1,-2,-3,-4,-5,-8,-9,-10,0,1,2]
    SNR = [-10]
    hash_distances_list = []
    for_image = False
    results = []
    lst = [0]
    for i in lst:
        DHD_path = f'./models_DHD/checkpoint.pth.tar'
        job_name = f'EVAL_retrival'
        evaluate_JSCC_path = f'./models/Exp2_hyparameter_{i}_SNR_10dB_0824.pth'
        evaluate_gan(args, args1, job_name, evaluate_JSCC_path, DHD_path,i,for_image)