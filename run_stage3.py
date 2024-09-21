import os

template = """
import numpy as np
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import lpips  # 导入LPIPS库
from torch.optim.lr_scheduler import CosineAnnealingLR
from source_DHD import *
from source_JSCC_MIMO import *
from datetime import datetime
import json

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

AugS = Augmentation(org_size, 1.0)
# AugT = Augmentation(org_size, 0.2)
AugT = Augmentation(org_size, 0.05)


Crop = nn.Sequential(Kg.CenterCrop(input_size))
Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

trainset = Loader(Img_dir, Train_dir, NB_CLS)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.DHD_batch_size, drop_last=True,
                                        shuffle=True, num_workers=args.DHD_num_workers)

Query_set = Loader(Img_dir, Query_dir, NB_CLS)
Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)
valid_loader = Query_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

######
#           DHD model Class Config (Discriminator)
######

def tensor_to_pil(tensor):
    # Convert a tensor to a PIL Image
    return transforms.ToPILImage()(tensor.cpu())

def load_checkpoint(model, filename='checkpoint.pth.tar'):
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

def get_discriminator(args,DHD_checkpoint):
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

    H = Hash_func(fc_dim, args.DHD_N_bits, NB_CLS=21)
    net = nn.Sequential(Baseline, H)
    net = nn.DataParallel(net)  # Add this line to wrap model for DataParallel

    _, best_mAP = load_checkpoint(net, filename=DHD_checkpoint)
    net.to(device)
    print(f"Loaded checkpoint with best mAP: {best_mAP}")

    return net, H
###########
#           JSCC model Class Config (Generator)
###########
def save_checkpoint(state, is_best, output_dir, filename):
    if is_best:
        T.save(state, os.path.join(output_dir, filename))  # save checkpoint
        print("=> Saving a new best model")

def get_generator(jscc_args, job_name,checkpoint_path):
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
    jscc_model.to(device)
    return jscc_model

def train_gan(discriminator_args, generator_args, job_name,JSCC_checkpoint,DHD_checkpoint):
    discriminator, H = get_discriminator(discriminator_args,DHD_checkpoint)
    generator = get_generator(generator_args, job_name,JSCC_checkpoint)
    
    # LPIPS计算模型
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Optimizers
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_args.DHD_init_lr, weight_decay=10e-6)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_args.JSCC_lr)
    
    # Schedulers
    d_scheduler = CosineAnnealingLR(d_optimizer, T_max=discriminator_args.DHD_max_epoch, eta_min=0)
    g_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lambda x: 0.8)
    es = EarlyStopping(mode='min', min_delta=0, patience=discriminator_args.JSCC_train_patience)

    HP_criterion = HashProxy(discriminator_args.DHD_temp)
    HD_criterion = HashDistill()
    REG_criterion = BCEQuantization(discriminator_args.DHD_std)

    num_classes = 21

    MAX_mAP = 0.0
    mAP = 0.0

    # AugT = Augmentation(256, discriminator_args.DHD_transformation_scale)  # Assuming org_size is 32 for CIFAR-10
    Crop = nn.Sequential(Kg.CenterCrop(224))
    Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

    n_critic = 1
    max_epoch = discriminator_args.DHD_max_epoch

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
        print(f'Epoch {epoch}/{max_epoch}')
        C_loss = 0.0
        S_loss = 0.0
        R_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            l1 = torch.tensor(0., device=device)
            l2 = torch.tensor(0., device=device)
            l3 = torch.tensor(0., device=device)
            if train_DHD == False:
                discriminator.eval()
                It = Norm(Crop(inputs))
                Xt = discriminator(It)
            else:
                discriminator.train()
                fake_inputs = generator(inputs, is_train=False).detach()  # detach to prevent gradients from flowing back to the generator
                Is = Norm(Crop(fake_inputs))
                It = Norm(Crop(inputs))

                Xt = discriminator(It)
                l1 = HP_criterion(Xt, H.P, labels)

                Xs = discriminator(Is)
                l2 = HD_criterion(Xs, Xt) * generator_args.DHD_lambda1
                l3 = REG_criterion(Xt) * generator_args.DHD_lambda2

                d_loss = l1 + l2 + l3
                d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)  # Retain graph to allow second backward pass
                d_optimizer.step()

                C_loss += l1.item()
                S_loss += l2.item()
                R_loss += l3.item()

                if (i+1) % 10 == 0:    # print every 10 mini-batches
                    print('[%3d] C: %.4f, S: %.4f, R: %.4f, mAP: %.4f, MAX mAP: %.4f' %
                        (i+1, C_loss / 10, S_loss / 10, R_loss / 10, mAP, MAX_mAP))
                    C_loss = 0.0
                    S_loss = 0.0
                    R_loss = 0.0
                
                if epoch >= generator_args.DHD_warm_up:
                    d_scheduler.step()
                discriminator.eval()

            # Train Generator
            if (train_DHD == False) or (train_DHD == True and epoch >= ONLY_DHD_train_epoch and i % n_critic == 0):
                g_optimizer.zero_grad()
                fake_inputs = generator(inputs, is_train=True)  # regenerate fake inputs with gradients
                reconstruction_loss = nn.MSELoss()(fake_inputs, inputs)
                HD_loss = HD_criterion(discriminator(Norm(Crop(fake_inputs))), Xt.detach())
                g_loss = reconstruction_loss + HD_loss * discriminator_args.DHD_lambda1
                g_loss.backward()
                g_optimizer.step()

                print("The Train Cosine Distance between Original and Transmitted Image is:", HD_loss.item())

                if (i + 1) % 100 == 0:
                    print(f'Step {i + 1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')


        print("++++++++++++ Start Validation ++++++++++++")
        valid_loss, valid_aux, val_hash, val_hamming, val_loss_with_DHD = validate_epoch(generator, discriminator, HD_criterion, discriminator_args, valid_loader, lpips_model)
        # valid_loss, valid_aux = validate_epoch(discriminator_args, valid_loader, generator, lpips_model)
        print("The Val Cosine Distance between Original and Transmitted Image is:", val_hash)
        print("The Val hamming Distance between Original and Transmitted Image is:", val_hamming)

        # g_scheduler.step()

        if epoch == 20:
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': discriminator.state_dict(),
            'best_mAP': MAX_mAP,
            'optimizer': d_optimizer.state_dict(),
            }, True, generator_args.DHD_output_dir, filename=job_name+"_only_train_DHD.pth.tar")

        if (epoch + 1) % discriminator_args.DHD_eval_epoch == 0 and (epoch + 1) >= discriminator_args.DHD_eval_init and train_DHD == True:
            mAP = DoRetrieval(device, discriminator.eval(), "./datasets/nuswide256", "./datasets/nuswide_DB.txt", "./datasets/nuswide_Query.txt", num_classes, 5000, discriminator_args)
            mAP_value = mAP.item()  # convert tensor to a Python number
            if mAP_value > MAX_mAP:
                MAX_mAP = mAP_value
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': discriminator.state_dict(),
                'best_mAP': MAX_mAP,
                'optimizer': d_optimizer.state_dict(),
                }, True, generator_args.DHD_output_dir, filename=job_name+".pth.tar")
                print("SAVE THE BEST CHECKPOINT of DHD")
                save_nets(job_name+"BestDHDmAP", generator, epoch)
                print("SAVE THE BEST CHECKPOINT of JSCC with highest DHD mAP")
            print("mAP: ", mAP_value, "MAX mAP: ", MAX_mAP)
        if train_DHD == True and epoch >= ONLY_DHD_train_epoch and i % n_critic == 0:
            # Save evaluation results in every epoch
            evaluation_metrics.append({
                'epoch': epoch,
                'valid_loss': valid_loss.item(),  
                'PSNR': valid_aux['psnr'], 
                'SSIM': valid_aux['ssim'],
                'LPIPS': valid_aux['lpips'],
                'DHD_MAXmap': MAX_mAP,
                'mAP': mAP_value,
                "HD_loss":HD_loss.item(),
                "val_hash":val_hash.item(),
                "val_hamming":val_hamming.item()
            })
        else:
            evaluation_metrics.append({
                'epoch': epoch,
                'valid_loss': valid_loss.item(),  
                'PSNR': valid_aux['psnr'], 
                'SSIM': valid_aux['ssim'],
                'LPIPS': valid_aux['lpips'],
                'DHD_MAXmap': MAX_mAP,
                'mAP': mAP_value,
                "HD_loss":"NOT Training JSCC",
                "val_hash":val_hash.item(),
                "val_hamming":val_hamming.item()
            })

        # Writing result into Json file
        with open(json_filename, 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)

        flag, best, best_epoch, bad_epochs = es.step(torch.tensor([val_loss_with_DHD]), epoch)
        if flag:
            print('ES criterion met; Keep training...')
            # print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
            # _ = load_weights(job_name, generator)
            # break
        else:
            if bad_epochs == 0:
                print('average l2_loss: ', valid_loss.item())
                save_nets(job_name, generator, epoch)
                best_epoch = epoch
                print('saving best net weights...')
            elif bad_epochs % (es.patience // 3) == 0:
                g_scheduler.step()
                print('lr updated: {:.5f}'.format(g_scheduler.get_last_lr()[0]))


def validate_epoch(generator, discriminator, HD_criterion, args, loader,lpips_model):
    generator.eval()
    #### 如果stage3 DHD也是要eval的
    if train_DHD == True:
        discriminator.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    lpips_hist = []
    hash_distance_list = []
    hamming_dist_list = []

    with torch.no_grad():
        with tqdm(loader, unit='batch') as tepoch:
            for _, (images, labels) in enumerate(tepoch):
                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                inputs_ori = images
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

                # output = generator(images, is_train=False)
                loss = nn.MSELoss()(JSCC_input, images)
                lpips_val = lpips_model(images, JSCC_input).mean()

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())
                epoch_postfix['lpips'] = '{:.4f}'.format(lpips_val.item())

                predictions = torch.chunk(JSCC_input, chunks=JSCC_input.size(0), dim=0)
                target = torch.chunk(images, chunks=images.size(0), dim=0)

                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())
                lpips_hist.append(lpips_val.item())
            
            loss_mean = np.nanmean(loss_hist)
            loss_new_mean = loss_mean + np.nanmean(hash_distance_list)

            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            lpips_mean = np.mean(lpips_hist)
            lpips_std = np.std(lpips_hist)

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            return_aux = {'psnr': psnr_mean,
                            'ssim': ssim_mean,
                            'lpips': lpips_mean,
                            'predictions': predictions,
                            'target': target,
                            'psnr_std': psnr_std,
                            'ssim_std': ssim_std,
                            'lpips_std': lpips_std}

            hash_distances = np.array(hash_distance_list)
            average_hash_distance = np.mean(hash_distances)

            hamming_dist_list = np.array(hamming_dist_list)
            average_hamming_dist = np.mean(hamming_dist_list)

    return loss_mean, return_aux, average_hash_distance, average_hamming_dist, loss_new_mean

if __name__ == '__main__':
    DHD_checkpoint = './models_DHD/checkpoint.pth.tar'
    ONLY_DHD_train_epoch = 20
    args.JSCC_P1 = {P_VALUE}
    args.JSCC_P2 = {P_VALUE}
    args.DHD_warm_up = 30
    args.JSCC_epoch = 40
    n_critic = 1
    args1 = args

    train_DHD = True
    if train_DHD:
        job_name = f'FINAL_40epoches_NEWStage3_TRAINJSCC_with_Lpips_{args.JSCC_P2}dB_SNR_0820_NOES'
        evaluate_JSCC_path1 = f"./models/NEWStage2_FINAL_TRAINJSCC_with_Lpips_{args.JSCC_P2}dB_SNR_0812.pth"
        train_gan(args, args1, job_name, evaluate_JSCC_path1, DHD_checkpoint)
    else:
        job_name = f'NEWStage2_FINAL_TRAINJSCC_with_Lpips_{args.JSCC_P2}dB_SNR_0812'
        evaluate_JSCC_path1 = None
        train_gan(args, args1, job_name, evaluate_JSCC_path1, DHD_checkpoint)


"""

def create_script(p_value):
    script_content = template.replace("{P_VALUE}", str(p_value))
    script_filename = f"FINAL_STAGE3_train_jscc_{p_value}_0821.py"
    with open(script_filename, "w") as script_file:
        script_file.write(script_content)
    return script_filename

def run_script(script_filename):
    os.system(f"python {script_filename}")

if __name__ == "__main__":
    for p_value in range(-10, -2):
        script_filename = create_script(p_value) 
        run_script(script_filename)
# -2