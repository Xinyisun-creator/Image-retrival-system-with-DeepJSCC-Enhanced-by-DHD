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
# Gallery_loader = torch.utils.data.DataLoader(Gallery_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)
Query_set = Loader(Img_dir, Query_dir, NB_CLS)
# Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)
Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=16, num_workers=args.DHD_num_workers)
valid_loader = Query_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

###########
#           Load DHD Model
###########

def load_checkpoint(model, filename):
    """Load checkpoint"""
    checkpoint = torch.load(filename)
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

def get_discriminator(args, discriminator_path):
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
    print("best_mAP of" + discriminator_path + ": " + str(best_mAP))

    return net, H

###########
#           Load JSCC Model
###########

def get_generator(jscc_args, job_name, checkpoint_path=None):
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
    if checkpoint_path is not None:
        # checkpoint_path = './models/train_JSCC_model_with_nuswide.pth'
        epoch = load_nets(checkpoint_path, jscc_model)
    return jscc_model

###########
#           Main Function
###########
def calculate_phash(image):
    return imagehash.phash(image)

def calculate_ahash(image):
    return imagehash.average_hash(image)

def calculate_dhash(image):
    return imagehash.dhash(image)

def tensor_to_pil(tensor):
    # Convert a tensor to a PIL Image
    return transforms.ToPILImage()(tensor.cpu())

def evaluate_gan(discriminator_args, generator_args, job_name, generator_load_path, discriminator_path):
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
    json_filename = f'LOG/{job_name}_evaluation_metrics_{timestamp}.json'
    evaluation_json_filename = f'LOG/{job_name}_final_evaluation_{timestamp}.json'

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

    return hash_distances, hamming_dist_list

def plot_hash_distances(hash_distances1, hash_distances2, hash_distances3, file_path):
    plt.figure(figsize=(12, 6))
    plt.hist(hash_distances1, bins=50, alpha=0.75, color='blue', label='Old JSCC')
    plt.hist(hash_distances2, bins=50, alpha=0.75, color='red', label='Stage1 JSCC')
    plt.hist(hash_distances3, bins=50, alpha=0.75, color='yellow', label='Stage2 JSCC')
    plt.title('Distribution of Hash Distances')
    plt.xlabel('Hash Distance')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(file_path)


def validate_epoch(loader, model, epoch):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    saved_images = False

    with torch.no_grad():
        with tqdm(loader, unit='batch') as tepoch:
            for _, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                output = model(images, is_train=False)
                loss = nn.MSELoss()(output, images)

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

                ######  Predictions  ######
                predictions = torch.chunk(output, chunks=output.size(0), dim=0)
                target = torch.chunk(images, chunks=images.size(0), dim=0)

                ######  PSNR/SSIM/etc  ######
                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                # Show the snr/loss/psnr/ssim
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())

                # Save one pair of original and output images for visualization
                if not saved_images:
                    save_tensor_as_image(images[0], f'original_image_epoch_{epoch}_with_new_loss.png')
                    save_tensor_as_image(output[0], f'output_image_epoch_{epoch}_with_new_loss.png')
                    saved_images = True
            
            loss_mean = np.nanmean(loss_hist)

            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            return_aux = {'psnr': psnr_mean,
                          'ssim': ssim_mean,
                          'predictions': predictions,
                          'target': target,
                          'psnr_std': psnr_std,
                          'ssim_std': ssim_std}

    return loss_mean, return_aux


def plot_hash_distances_per_data_point(hash_distances1, hash_distances2, hash_distances3, file_path, mode):
    plt.figure(figsize=(12, 6))

    # Number of data points
    num_data_points = len(hash_distances1)

    # Plot scatter plot for each model
    plt.scatter(range(num_data_points), hash_distances1, alpha=0.75, color='blue', label='Old JSCC')
    plt.scatter(range(num_data_points), hash_distances2, alpha=0.75, color='red', label='Stage1 JSCC')
    plt.scatter(range(num_data_points), hash_distances3, alpha=0.75, color='yellow', label='Stage2 JSCC')

    if mode == "hash":
        plt.title('Cosine Hash Distances for Each Data Point')
        plt.xlabel('Data Point Index')
        plt.ylabel('Cosine Hash Distance')
        plt.legend(loc='upper right')
    
    if mode == "hamming":
        plt.title('Hamming Distances for Each Data Point')
        plt.xlabel('Data Point Index')
        plt.ylabel('Hamming Distance')
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()

def evaluate_multiple_jscc_models(jscc_paths, discriminator_path, job_name):
    results = {}
    for snr, jscc_path in jscc_paths.items():
        hash_dist, hamming_dist = evaluate_gan(args, args, job_name, jscc_path, discriminator_path)
        results[snr] = {
            "hash_distances": hash_dist.tolist(),
            "average_hash_distance": np.mean(hash_dist).item(),
            "hamming_distances": hamming_dist.tolist(),
            "average_hamming_distance": np.mean(hamming_dist).item()
        }
    
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f'evaluation_results_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {json_filename}")

    return results

if __name__ == '__main__':
    job_name = 'JSCC_model_evaluate_nuswide_with_new_loss'

    jscc_paths = {}
    for snr in range(-10, 14):  # SNR range from -10 to 13
        if snr < -4:
            jscc_paths[f'{snr}dB'] = f'./models/Stage2_TRAINJSCC_with_Lpips_{snr}dB_SNR_0709.pth'
        else:
            jscc_paths[f'{snr}dB'] = f'./models/Stage2_TRAINJSCC_with_Lpips_{snr}dB_SNR_0709_valDH.pth'

    DHD_path = './models_DHD/checkpoint.pth.tar'
    results = evaluate_multiple_jscc_models(jscc_paths, DHD_path, job_name)

    # Plotting the results
    for metric in ['hash_distances', 'hamming_distances']:
        all_distances = [results[snr][metric] for snr in jscc_paths.keys()]
        plot_hash_distances_per_data_point(
            *all_distances,
            f'{metric}_per_data_point.png',
            mode='hash' if metric == 'hash_distances' else 'hamming'
        )
