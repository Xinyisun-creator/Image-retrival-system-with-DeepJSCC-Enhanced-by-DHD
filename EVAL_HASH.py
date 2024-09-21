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
    generator = get_generator(generator_args, job_name, generator_load_path)

    discriminator.eval()
    generator.eval()
    
    # Optimizer for JSCC
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_args.JSCC_lr)
    
    # Scheduler for JSCC
    g_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lambda x: 0.8)
    es = EarlyStopping(mode='min', min_delta=0, patience=generator_args.JSCC_train_patience)

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


    print("Processing query data...")
    for i, data in enumerate(tqdm(Query_loader, desc="Query")):
        inputs_ori, labels = data[0].to(device), data[1].to(device)
        inputs = Norm(Crop(inputs_ori))

        inputs_ori = inputs_ori.to(device).float()
        JSCC_input = generator(inputs_ori, is_train=False).detach()
        # if for_image:
        #     save_tensor_as_jpg(JSCC_input, './output_images', f'EX2_Stage2_MYGPU_{SNR}')
        #     return "Done"

        JSCC_outputs = Norm(Crop(JSCC_input))
        original_Hash = discriminator(inputs)

        JSCC_Hash = discriminator(JSCC_outputs)
        hash_distance_list.append(HD_criterion(original_Hash, JSCC_Hash).cpu().detach().numpy())

        binary_JSCC = torch.sign(JSCC_Hash)
        binary_original = torch.sign(original_Hash)

        small_hamming = []
        for j in range(binary_JSCC.shape[0]):
            hamming_dist = (binary_JSCC.shape[1] - binary_JSCC[j, :] @ binary_original[j,:].t())
            small_hamming.append(hamming_dist.item())
        hamming_dist_list.append(np.mean(np.array((small_hamming))))
    
    hash_distances = np.array(hash_distance_list)
    average_hash_distance = np.mean(hash_distances)

    hamming_dist_list = np.array(hamming_dist_list)
    average_hamming_dist = np.mean(hamming_dist_list)

    # Evaluation metrics
    print("average_hash_distance: ", average_hash_distance)
    print("average_hamming_dist: ", average_hamming_dist)

    # Convert float32 to float for JSON serialization
    evaluation_metrics.append({
        'average_hash_distance': float(average_hash_distance),
        'average_hamming_dist': float(average_hamming_dist),
        'hash_distances': hash_distances.tolist(),
        'hamming_distances': hamming_dist_list.tolist()
    })
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    # writw json file
    with open(json_filename, 'w') as f:
        json.dump(evaluation_metrics, f)

    return hash_distances,hamming_dist_list

def plot_snr_performance(snr_list, hash_distances, file_path):
    """
    绘制不同 SNR 下的哈希距离表现

    参数:
    snr_list (list): SNR 值的列表
    hash_distances (list): 每个 SNR 对应的哈希距离列表
    file_path (str): 保存图像的路径
    """
    plt.figure(figsize=(15, 10))

    num_snr = len(snr_list)
    for i in range(num_snr):
        plt.subplot(num_snr, 1, i + 1)
        plt.hist(hash_distances[i], bins=50, alpha=0.75, label=f'SNR: {snr_list[i]} dB')
        plt.title(f'Distribution of Hash Distances at SNR: {snr_list[i]} dB')
        plt.xlabel('Hash Distance')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()

if __name__ == '__main__':
    args1 = args
    # SNR = [-1,-2,-3,-4,-5,-8,-9,-10,0,1,2]
    SNR = [-10]
    hash_distances_list = []
    for_image = False
    results = []
    lst = [0.04,0.22,0.24]
    for i in lst:
        DHD_path = f'0904_Exp2_Stage3_hyparameter_{i}_SNR_10dB_0827.pth.tar'
        job_name = f'Eval_Stage3_MIMO_vs_Stage3_DHD_SNR_{i}dB_bestmAP'
        print(f"Test performance for Baseline g SNR: {i}")
        evaluate_JSCC_path = f'./models/Exp2_hyparameter_{i}_SNR_10dB_0824.pth'
        if for_image:
            evaluate_gan(args, args1, job_name, evaluate_JSCC_path, DHD_path,i,for_image)
        else:
            hash_dist, hamming_dist = evaluate_gan(args, args1, job_name, evaluate_JSCC_path, DHD_path,i,for_image)
        # Append the results for this SNR to the list
        results.append([i, hash_dist, hamming_dist])
    # Write the results to a CSV file
    if not for_image:
        with open('./CSV_ExperimentResult/0905.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(['SNR', 'hash_dist', 'hamming_dist'])
            # Write the data
            writer.writerows(results)

    # for i in range(11,14):
    #     DHD_path = './models_DHD/checkpoint.pth.tar'
    #     job_name = f'Eval_Stage3_MIMO_vs_Stage3_DHD_SNR_{i}dB_bestmAP'
    #     print(f"Test performance for Baseline in SNR: {i}")
    #     evaluate_JSCC_path = f'./models/0830_JSCCBASELINE_{i}_evaluation_metrics.pth'
    #     if for_image:
    #         evaluate_gan(args, args1, job_name, evaluate_JSCC_path, DHD_path,i,for_image)
    #     else:
    #         hash_dist, hamming_dist = evaluate_gan(args, args1, job_name, evaluate_JSCC_path, DHD_path,i,for_image)
    #     # Append the results for this SNR to the list
    # #     results.append([i, hash_dist, hamming_dist])
    # # # Write the results to a CSV file
    # # if not for_image:
    # #     with open('./CSV_ExperimentResult/Baseline_JSCC_11_13.csv', 'w', newline='') as f:
    # #         writer = csv.writer(f)
    # #         # Write the header
    # #         writer.writerow(['SNR', 'hash_dist', 'hamming_dist'])
    # #         # Write the data
    # #         writer.writerows(results)