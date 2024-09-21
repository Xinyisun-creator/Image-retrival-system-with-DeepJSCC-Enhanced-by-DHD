import numpy as np
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import lpips

from source_DHD import *
from source_JSCC_MIMO import *

from datetime import datetime
import json
import argparse

#########
#           Parameter Setting
#########
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
dname = 'nuswide'
path = './datasets/'
args = get_args_parser()
args.JSCC_P1 = {P_VALUE}
args.JSCC_P2 = {P_VALUE}

Img_dir = path + dname + '256'
Train_dir = path + dname + '_Train.txt'
Gallery_dir = path + dname + '_DB.txt'
Query_dir = path + dname + '_Query.txt'
org_size = 256
input_size = 224
NB_CLS = 21

trainset = Loader(Img_dir, Train_dir, NB_CLS)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.JSCC_train_batch_size, drop_last=True, shuffle=True, num_workers=12)

Query_set = Loader(Img_dir, Query_dir, NB_CLS)
Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.JSCC_val_batch_size, num_workers=12)
valid_loader = Query_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

###########
#           JSCC model Class Config
###########

def get_generator(jscc_args, job_name):
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
    return jscc_model.to(jscc_args.device)

def train_gan(generator_args, job_name):
    generator = get_generator(generator_args, job_name)
    
    # Optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_args.JSCC_lr)
    
    # Scheduler
    g_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lambda x: 0.8)
    es = EarlyStopping(mode='min', min_delta=0, patience=generator_args.JSCC_train_patience)

    num_classes = 21

    # evaluation metrics for JSON file
    evaluation_metrics = []

    # time stamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_filename = f'LOG/{job_name}_evaluation_metrics_{timestamp}.json'
    evaluation_json_filename = f'LOG/{job_name}_final_evaluation_{timestamp}.json'

    # Initial epoch
    epoch = 0

    while epoch < generator_args.JSCC_epoch and not generator_args.JSCC_resume:
        epoch += 1
        print(f'Epoch {epoch}/{generator_args.JSCC_epoch}')
        R_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            g_optimizer.zero_grad()
            fake_inputs = generator(inputs, is_train=True)
            reconstruction_loss = nn.MSELoss()(fake_inputs, inputs)
            reconstruction_loss.backward()
            g_optimizer.step()

            R_loss += reconstruction_loss.item()

            if (i + 1) % 500 == 0:
                print(f'Step {i + 1}, G Loss: {reconstruction_loss.item()}')

        valid_loss, valid_aux = validate_epoch(generator_args, valid_loader, generator)

        # g_scheduler.step()

        # Save evaluation results in every epoch
        evaluation_metrics.append({
            'epoch': epoch,
            'valid_loss': valid_loss.item(),  
            'PSNR': valid_aux['psnr'], 
            'SSIM': valid_aux['ssim'],
            'LPIPS': valid_aux['lpips'],  # Add LPIPS to the evaluation metrics
        })

        # Writing result into Json file
        with open(json_filename, 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)

        flag, best, best_epoch, bad_epochs = es.step(torch.tensor([valid_loss]), epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
            _ = load_weights(job_name, generator)
            break
        else:
            if bad_epochs == 0:
                print('average l2_loss: ', valid_loss.item())
                save_nets(job_name, generator, epoch)
                best_epoch = epoch
                print('saving best net weights...')
            elif bad_epochs % (es.patience // 3) == 0:
                g_scheduler.step()
                print('lr updated: {:.5f}'.format(g_scheduler.get_last_lr()[0]))

    print('evaluating...')
    print(job_name)

    final_evaluation_metrics = []

    # for P in range(6, 16, 2):
    #     generator.P1, generator.P2 = P, P
    #     _, eval_aux = validate_epoch(discriminator_args, eval_loader, generator)
    #     print(eval_aux['psnr'])
    #     print(eval_aux['ssim'])
        
    #     final_evaluation_metrics.append({
    #         'psnr': eval_aux['psnr'],
    #         'ssim': eval_aux['ssim'],
    #     })

    # with open(evaluation_json_filename, 'w') as f:
    #     json.dump(final_evaluation_metrics, f, indent=4)

def validate_epoch(args, loader, model):
    # Initialize LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(args.device)

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    lpips_hist = []

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

                ######  PSNR/SSIM/LPIPS  ######
                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                lpips_vals = [lpips_fn(pred, tgt).item() for pred, tgt in zip(predictions, target)]
                lpips_hist.extend(lpips_vals)
                epoch_postfix['lpips'] = torch.mean(torch.tensor(lpips_vals)).item()

                # Show the snr/loss/psnr/ssim/lpips
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())
            
            loss_mean = np.nanmean(loss_hist)

            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            lpips_hist = torch.tensor(lpips_hist)
            lpips_mean = torch.mean(lpips_hist).item()
            lpips_std = torch.sqrt(torch.var(lpips_hist)).item()

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            return_aux = {'psnr': psnr_mean,
                          'ssim': ssim_mean,
                          'lpips': lpips_mean,  # Add LPIPS to the return values
                          'predictions': predictions,
                          'target': target,
                          'psnr_std': psnr_std,
                          'ssim_std': ssim_std,
                          'lpips_std': lpips_std}

    return loss_mean, return_aux

if __name__ == '__main__':
    job_name = f"0830_JSCCBASELINE_{args.JSCC_P2}_evaluation_metrics"
    args1 = args
    train_gan(args1, job_name)