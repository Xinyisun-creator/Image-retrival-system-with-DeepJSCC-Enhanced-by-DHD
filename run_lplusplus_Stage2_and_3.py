import numpy as np 
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR

from source_DHD import *
from source_JSCC_lplusplus import *

from datetime import datetime
import json

import pdb


#########
#           Parameter Setting
#########

dname = 'nuswide'
path = './datasets/'
args = get_args()

Img_dir = path + dname + '256'
Train_dir = path + dname + '_Train.txt'
Gallery_dir = path + dname + '_DB.txt'
Query_dir = path + dname + '_Query.txt'
org_size = 256
input_size = 224
NB_CLS = 21

AugS = Augmentation(org_size, 1.0)
AugT = Augmentation(org_size, 0.2)

Crop = nn.Sequential(Kg.CenterCrop(input_size))
Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

trainset = Loader(Img_dir, Train_dir, NB_CLS)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.DHD_batch_size, drop_last=True,
                                        shuffle=True, num_workers=args.DHD_num_workers)

Query_set = Loader(Img_dir, Query_dir, NB_CLS)
Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)
valid_loader = Query_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

writter = SummaryWriter('./source_JSCC_lplusplus/runs/' + job_name)

######
#           DHD model Class Config (DHD)
######

def load_checkpoint(model, filename='checkpoint.pth.tar'):
    """Load checkpoint"""
    checkpoint = T.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint['epoch'], checkpoint['best_mAP']

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

def get_DHD(args, DHD_checkpoint):
    if args.DHD_encoder == 'AlexNet':
        Baseline = AlexNet()
        fc_dim = 4096
    elif args.DHD_encoder == 'ResNet':
        Baseline = ResNet()
        fc_dim = 2048
    elif args.DHD_encoder == 'ViT':
        Baseline = ViT('vit_base_patch16_224')
        fc_dim = 768
    elif args.DHD_encoder == 'DeiT':
        Baseline = DeiT('deit_base_distilled_patch16_224')
        fc_dim = 768
    elif args.DHD_encoder == 'SwinT':
        Baseline = SwinT('swin_base_patch4_window7_224')
        fc_dim = 1024
    else:
        print("Wrong encoder name.")
        return None

    H = Hash_func(fc_dim, args.DHD_N_bits, NB_CLS=21).to(args.device)  # 确保Hash_func移动到设备
    net = nn.Sequential(Baseline.to(args.device), H).to(args.device)  # 确保Baseline和H移动到设备
    net = nn.DataParallel(net)  # 然后包装成DataParallel
    net.to(args.device)  # 确保DataParallel的模型在指定设备上

    _, best_mAP = load_checkpoint(net, filename=DHD_checkpoint)
    print(f"Loaded checkpoint with best mAP: {best_mAP}")

    return net, H


###########
#           JSCC model Class Config (JSCC)
###########
def save_checkpoint(state, is_best, output_dir, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        T.save(state, os.path.join(output_dir, filename))  # save checkpoint
        print("=> Saving a new best model")

def get_JSCC(args, job_name, checkpoint_path):
    ###### The JSCC Model using Swin Transformer ######
    enc_kwargs = dict(
        args = args, n_trans_feat = args.n_trans_feat, img_size=(args.image_dims[0], args.image_dims[1]),
        embed_dims=[args.embed_size, args.embed_size], depths=[args.depth[0], args.depth[1]], num_heads=[args.n_heads, args.n_heads],
        window_size=args.window_size, mlp_ratio=args.mlp_ratio, qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )

    dec_kwargs = dict(
        args = args, n_trans_feat = args.n_trans_feat, img_size=(args.image_dims[0], args.image_dims[1]),
        embed_dims=[args.embed_size, args.embed_size], depths=[args.depth[1], args.depth[0]], num_heads=[args.n_heads, args.n_heads],
        window_size=args.window_size, mlp_ratio=args.mlp_ratio, norm_layer=nn.LayerNorm, patch_norm=True
    )
    source_enc = Swin_Encoder(**enc_kwargs).to(args.device)
    source_dec = Swin_Decoder(**dec_kwargs).to(args.device)
    jscc_model = Swin_JSCC(args, source_enc, source_dec)
    jscc_model = nn.DataParallel(jscc_model)


    # # load pre-trained
    # if args.resume:
    #     _ = load_weights(job_name, jscc_model)

    # jscc_model = nn.DataParallel(jscc_model).to(args.device)  # 确保DataParallel的模型在指定设备上

    return jscc_model

def dynamic_weight_adaption(current):
    # dynamically change the weight; perform it every epoch
    # target -> separate trained PSNR; current -> current PSNR of adaptive model
    delta = TARGET - current
    for i in range(len(delta)):
        if delta[i] <= args.threshold:
            weight[i] = 0
        else:
            weight[i] = 2**(args.alpha*delta[i])

    clipped_weight = np.clip(weight, args.min_clip, args.max_clip)
    return clipped_weight
    
def train_gan(args, job_name, JSCC_checkpoint, DHD_checkpoint, weight):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.DHD_gpu_id
    DHD, H = get_DHD(args, DHD_checkpoint)
    JSCC = get_JSCC(args, job_name, JSCC_checkpoint)

    # Optimizers
    d_optimizer = torch.optim.Adam(DHD.parameters(), lr=args.DHD_init_lr, weight_decay=10e-6)
    j_optimizer = torch.optim.Adam(JSCC.parameters(), lr=args.lr)

    # Schedulers
    d_scheduler = CosineAnnealingLR(d_optimizer, T_max=args.DHD_max_epoch, eta_min=0)
    j_scheduler = LS.MultiplicativeLR(j_optimizer, lr_lambda=lambda x: 0.9)
    es = EarlyStopping(mode='min', min_delta=0, patience=args.train_patience)

    # TARGET = np.array([24.75, 27.85, 30.1526917, 32.01, 33.2777652, 34.55393814])
    TARGET = np.array([34.55393814])

    HP_criterion = HashProxy(args.DHD_temp).to(args.device)
    HD_criterion = HashDistill().to(args.device)
    REG_criterion = BCEQuantization(args.DHD_std).to(args.device)

    num_classes = 21

    MAX_mAP = 0.0
    mAP = 0.0

    AugT = Augmentation(256, args.DHD_transformation_scale)  # Assuming org_size is 32 for CIFAR-10
    Crop = nn.Sequential(Kg.CenterCrop(224)).to(args.device)
    Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]).to(args.device), std=torch.as_tensor([0.229, 0.224, 0.225]).to(args.device)))

    n_critic = 1
    max_epoch = args.DHD_max_epoch

    # evaluation metrics for JSON file
    evaluation_metrics = []

    # time stamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_filename = f'LOG/{job_name}_evaluation_metrics_{timestamp}.json'
    evaluation_json_filename = f'LOG/{job_name}_final_evaluation_{timestamp}.json'

    # Initial epoch
    epoch = 0

    while epoch < args.epoch and not args.resume:
        epoch += 1
        print(f'Epoch {epoch}/{max_epoch}')
        C_loss = 0.0
        S_loss = 0.0
        R_loss = 0.0

        with tqdm(train_loader, unit='batch') as tepoch:
            for i, data in enumerate(tepoch):
                inputs, labels = data[0].to(device), data[1].to(device)
                if train_DHD == False:
                    DHD.eval()
                else:
                    DHD.train()

                # Train DHD
                bw = np.random.randint(args.min_trans_feat, args.max_trans_feat+1)
                JSCC.eval()
                fake_inputs = JSCC(inputs, bw).detach()  # detach to prevent gradients from flowing back to the JSCC

                l1 = torch.tensor(0., device=device)
                l2 = torch.tensor(0., device=device)
                l3 = torch.tensor(0., device=device)

                Is = Norm(Crop(fake_inputs))
                It = Norm(Crop(inputs))

                Xt = DHD(It)
                l1 = HP_criterion(Xt, H.P, labels)

                Xs = DHD(Is)
                l2 = HD_criterion(Xs, Xt) 
                l3 = REG_criterion(Xt) 

                if train_DHD:
                    d_loss = l1 + l2 + l3
                    d_optimizer.zero_grad()
                    d_loss.backward(retain_graph=True)  # Retain graph to allow second backward pass
                    d_optimizer.step()

                C_loss += l1.item()
                S_loss += l2.item()
                R_loss += l3.item()

                # Train JSCC
                if i % n_critic == 0:
                    JSCC.train()

                    epoch_postfix = OrderedDict()

                    images = inputs.to(args.device).float()

                    j_optimizer.zero_grad()

                    fake_inputs = JSCC(images, bw)  # regenerate fake inputs with gradients
                    
                    reconstruction_loss = nn.MSELoss()(fake_inputs, images) * weight[bw-args.min_trans_feat]

                    HD_loss = HD_criterion(DHD(Norm(Crop(fake_inputs))), Xt.detach())
                    JSCC_loss = reconstruction_loss + HD_loss * args.DHD_lambda1
                    JSCC_loss.backward()
                    j_optimizer.step()

                    epoch_postfix['l2_loss'] = '{:.4f}'.format(JSCC_loss.item())
                    tepoch.set_postfix(**epoch_postfix)

                if (i + 1) % 500 == 0:
                    print(f'Step {i + 1}, D Loss: {d_loss.item()}, J Loss: {JSCC_loss.item()}')

        valid_loss, valid_aux = validate_epoch(valid_loader, JSCC)
        print("The Cosine Distance between Original and Transmitted Image is:", HD_loss.item())

        writter.add_scalar('loss', valid_loss, epoch)
        writter.add_scalar('psnr', valid_aux['psnr'], epoch)

        if epoch % args.freq == 0:
            current_psnr = np.array([0.0 for _ in range(args.min_trans_feat, args.max_trans_feat+1)])
            for i in range(len(weight)):
                _, valid_aux = validate_epoch(valid_loader, JSCC, args.min_trans_feat + i, True)
                current_psnr[i] = valid_aux['psnr']

            weight = dynamic_weight_adaption(current_psnr)
            
            # writter.add_scalars('all_psnr', {'bw1': current_psnr[0], 'bw2': current_psnr[1], 'bw3': current_psnr[2],
            #                                  'bw4': current_psnr[3], 'bw5': current_psnr[4], 'bw6': current_psnr[5]}, epoch)
            # writter.add_scalars('weights', {'weight1': weight[0], 'weight2': weight[1], 'weight3': weight[2],
            #                                 'weight4': weight[3], 'weight5': weight[4], 'weight6': weight[5]}, epoch)

            writter.add_scalars('all_psnr', {'bw1':current_psnr[0]}, epoch)
            writter.add_scalars('weights', {'weight1':weight[0]}, epoch)  


        if (epoch + 1) % args.DHD_eval_epoch == 0 and (epoch + 1) >= args.DHD_eval_init and train_DHD == True:
            mAP = DoRetrieval(device, DHD.eval(), "./datasets/nuswide256", "./datasets/nuswide_DB.txt", "./datasets/nuswide_Query.txt", num_classes, 5000, args)
            mAP_value = mAP.item()
            if mAP_value > MAX_mAP:
                MAX_mAP = mAP_value
                save_checkpoint({'epoch': epoch + 1, 'state_dict': DHD.state_dict(), 'best_mAP': MAX_mAP, 'optimizer': d_optimizer.state_dict()},
                                True, args.DHD_output_dir, filename=job_name + ".pth.tar")
                print("SAVE THE BEST CHECKPOINT of DHD")
            print("mAP: ", mAP_value, "MAX mAP: ", MAX_mAP)

        evaluation_metrics.append({'epoch': epoch, 'valid_loss': valid_loss.item(), 'PSNR': valid_aux['psnr'],
                                   'SSIM': valid_aux['ssim'], 'DHD_MAXmap': MAX_mAP, 'mAP': mAP_value if train_DHD else "NotTrainingDHD",
                                   "HD_loss": HD_loss.item()})

        with open(json_filename, 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)

        flag, best, best_epoch, bad_epochs = es.step(torch.Tensor([valid_loss]), epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
            _ = load_weights(job_name, JSCC)
            break
        else:
            if bad_epochs == 0:
                print('average l2_loss: ', valid_loss.item())
                save_nets(job_name, JSCC, epoch)
                best_epoch = epoch
                print('saving best net weights...')
            elif bad_epochs % 20 == 0:
                j_scheduler.step()
                print('lr updated: {:.5f}'.format(j_scheduler.get_last_lr()[0]))

    # Save final epoch checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': DHD.state_dict(),
        'best_mAP': MAX_mAP,
        'optimizer': d_optimizer.state_dict(),
        }, True, args.DHD_output_dir, filename=job_name+"_last_epoch.pth.tar")
    print("SAVE FINAL EPOCH CHECKPOINT of DHD")

def validate_epoch(loader, model, bw=args.trg_trans_feat, disable=False):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    #msssim_hist = []
    power = []

    with torch.no_grad():
        with tqdm(loader, unit='batch', disable=disable) as tepoch:
            for _, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                output = model(images, bw, snr=args.link_qual)
                
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

if __name__ == '__main__':
    weight = np.array([1.0 for _ in range(args.min_trans_feat, args.max_trans_feat+1)])

    DHD_checkpoint = './models_DHD/checkpoint.pth.tar'

    train_DHD = True
    job_name = "TEST"+ 'JSCC_swin_adapt_lr_' + args.channel_mode +'_dataset_'+str(args.dataset) + '_link_qual_' + str(args.link_qual) + '_n_trans_feat_' + str(args.n_trans_feat)\
             + '_hidden_size_' + str(args.hidden_size) + '_n_heads_' + str(args.n_heads) + '_n_layers_' + str(args.n_layers) +'_is_adapt_'+ str(args.adapt)
    if args.adapt:
        job_name = "TEST"+ job_name + '_link_rng_' + str(args.link_rng)  + '_min_trans_feat_' + str(args.min_trans_feat) + '_max_trans_feat_' + str(args.max_trans_feat) + \
                    '_unit_trans_feat_' + str(args.unit_trans_feat) + '_trg_trans_feat_' + str(args.trg_trans_feat) 

    evaluate_JSCC_path1 = './models_JSCC/Stage2_TRAINJSCC_with_FixedDHD_20dB_SNR_0709.pth'
    try:
        train_gan(args, job_name, evaluate_JSCC_path1, DHD_checkpoint, weight)
    except Exception as e:
        print(f"Exception occurred: {e}")
        pdb.post_mortem()  # Enter post-mortem debugging
