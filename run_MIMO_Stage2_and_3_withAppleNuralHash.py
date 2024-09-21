import numpy as np
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import lpips  # Import LPIPS library
from torch.optim.lr_scheduler import CosineAnnealingLR
from source_DHD import *  # Import custom DHD model components
from source_JSCC_MIMO import *  # Import JSCC-MIMO components
from datetime import datetime
import json
import onnxruntime
from PIL import Image
from torchvision import transforms

# Define preprocessing transformations
resize_and_normalize = transforms.Compose([
    transforms.Resize((360, 360)),  # Resize to 360x360
    transforms.ToTensor(),
    # Uncomment if needed: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#########
# Parameter Setting
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

# Data augmentation
AugS = Augmentation(org_size, 1.0)
AugT = Augmentation(org_size, 0.2)

# Preprocessing layers
Crop = nn.Sequential(Kg.CenterCrop(input_size))
Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

# Load datasets
trainset = Loader(Img_dir, Train_dir, NB_CLS)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.DHD_batch_size, drop_last=True,
                                           shuffle=True, num_workers=args.DHD_num_workers)

Query_set = Loader(Img_dir, Query_dir, NB_CLS)
Query_loader = torch.utils.data.DataLoader(Query_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)
valid_loader = Query_loader

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

######
# Load ONNX Model
######

# Load ONNX model
onnx_session = onnxruntime.InferenceSession("./model_Apple_ONNX_neuralhash/model.onnx")

# Load output hash matrix
seed1 = open("./model_Apple_ONNX_neuralhash/neuralhash_128x96_seed1.dat", 'rb').read()[128:]
seed1 = np.frombuffer(seed1, dtype=np.float32)

# Change from 96 to 64
seed1 = seed1.reshape([96, 128])  # Adjust to 64 instead of 96

def onnx_inference(image_array):
    """
    Run the ONNX model inference on a preprocessed image.
    """
    # Preprocess image array to match ONNX model input requirements
    arr = image_array.astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

    # Run model inference
    inputs = {onnx_session.get_inputs()[0].name: arr}
    outs = onnx_session.run(None, inputs)

    # Calculate hash output using seed matrix
    hash_output = seed1.dot(outs[0].flatten())
    return hash_output

def z_score_normalize(tensor):
    """
    Perform Z-score normalization on a PyTorch tensor.
    Args:
        tensor: PyTorch tensor to be normalized.
    Returns:
        Z-score normalized tensor.
    """
    mean_val = torch.mean(tensor)
    std_dev = torch.std(tensor)
    # Avoid division by zero in case of zero std deviation
    if std_dev == 0:
        return tensor
    normalized_tensor = (tensor - mean_val) / std_dev
    return normalized_tensor

def preprocess_images(image_tensor):
    """
    Preprocesses a batch of images by resizing and normalizing.
    Args:
        image_tensor: A batch of images in torch tensor format.
    Returns:
        A batch of preprocessed images.
    """
    preprocessed_images = []
    for img in image_tensor:
        img_pil = transforms.ToPILImage()(img.cpu())  # Convert tensor to PIL image
        img_resized = resize_and_normalize(img_pil)   # Resize and normalize
        preprocessed_images.append(img_resized)
    return torch.stack(preprocessed_images).to(device)

def onnx_hash(image_tensor):
    """
    Calculate ONNX hash for a batch of images.
    Args:
        image_tensor: A batch of images in torch tensor format.
    Returns:
        A batch of hash outputs as a PyTorch tensor.
    """
    hash_outputs = []

    # Process each image in the batch
    for img in image_tensor:
        # Detach and convert image tensor to numpy array
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format

        # Run ONNX inference
        hash_output = onnx_inference(img_np)
        hash_outputs.append(hash_output)

    # Convert the list of numpy arrays into a single numpy array
    hash_outputs_np = np.array(hash_outputs)

    # Convert the numpy array to a torch tensor
    hash_outputs_tensor = torch.from_numpy(hash_outputs_np).float().to(device)
    
    return hash_outputs_tensor

def validate_HD(generator, HD_criterion):
    hash_distance_list = []
    hamming_dist_list = []
    for i, data in enumerate(tqdm(Query_loader, desc="Query")):
        inputs_ori, labels = data[0].to(device), data[1].to(device)
        inputs = Norm(Crop(inputs_ori))
        inputs_ori = inputs_ori.to(device).float()
        JSCC_input = generator(inputs_ori, is_train=False).detach()
        JSCC_outputs = Norm(Crop(JSCC_input))

        # Use ONNX model for hash generation
        original_Hash = onnx_hash(inputs)
        JSCC_Hash = onnx_hash(JSCC_outputs)

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

    return average_hash_distance, average_hamming_dist

###########
# JSCC model Class Config (Generator)
###########
def save_checkpoint(state, is_best, output_dir, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        T.save(state, os.path.join(output_dir, filename))  # save checkpoint
        print("=> Saving a new best model")

def get_generator(jscc_args, job_name, checkpoint_path):
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
    return jscc_model.to(jscc_args.device)

def train_gan(discriminator_args, generator_args, job_name, JSCC_checkpoint, DHD_checkpoint):
    generator = get_generator(generator_args, job_name, JSCC_checkpoint)
    
    # LPIPS calculation model
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_args.JSCC_lr)
    
    # Schedulers
    g_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lambda x: 0.8)
    es = EarlyStopping(mode='min', min_delta=0, patience=discriminator_args.JSCC_train_patience)

    HP_criterion = HashProxy(discriminator_args.DHD_temp)
    HD_criterion = HashDistill()
    REG_criterion = BCEQuantization(discriminator_args.DHD_std)

    num_classes = 21

    MAX_mAP = 0.0
    mAP = 0.0

    AugT = Augmentation(256, discriminator_args.DHD_transformation_scale)
    Crop = nn.Sequential(Kg.CenterCrop(224))
    Norm = nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

    n_critic = 1
    max_epoch = discriminator_args.DHD_max_epoch

    # Evaluation metrics for JSON file
    evaluation_metrics = []

    # Time stamp
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
            inputs_resized = preprocess_images(inputs)
            JSCC_input = generator(inputs_resized, is_train=False).detach()

            l1 = torch.tensor(0., device=device)
            l2 = torch.tensor(0., device=device)
            l3 = torch.tensor(0., device=device)

            # Use ONNX model for hash generation
            Xt_raw = onnx_hash(inputs_resized)  # Raw hash output
            Xs_raw = onnx_hash(JSCC_input)      # Raw hash output for JSCC input

            # Z-score normalization
            Xt = z_score_normalize(Xt_raw)
            Xs = z_score_normalize(Xs_raw)

            l2 = HD_criterion(Xs, Xt) 
            S_loss += l2.item()

            if i % n_critic == 0:
                g_optimizer.zero_grad()
                fake_inputs = generator(inputs_resized, is_train=True)  # regenerate fake inputs with gradients
                reconstruction_loss = nn.MSELoss()(fake_inputs, inputs_resized)

                # Detach fake_inputs before using in onnx_hash
                HD_loss = HD_criterion(onnx_hash(fake_inputs.detach()), Xt.detach())
                g_loss = reconstruction_loss + HD_loss * discriminator_args.DHD_lambda1
                g_loss.backward()
                g_optimizer.step()

            if (i + 1) % 500 == 0:
                print(f'Step {i + 1}, G Loss: {g_loss.item()}')

        print("++++++++++++ Start Validation ++++++++++++")
        val_hash, val_hamming = validate_HD(generator, HD_criterion)
        valid_loss, valid_aux = validate_epoch(discriminator_args, valid_loader, generator, lpips_model)
        print("The Train Cosine Distance between Original and Transmitted Image is:", HD_loss.item())
        print("The Val Cosine Distance between Original and Transmitted Image is:", val_hash)
        print("The Val hamming Distance between Original and Transmitted Image is:", val_hamming)

        g_scheduler.step()

        if (epoch + 1) % discriminator_args.DHD_eval_epoch == 0 and (epoch + 1) >= discriminator_args.DHD_eval_init:
            mAP = DoRetrieval(device, generator, "./datasets/nuswide256", "./datasets/nuswide_DB.txt", "./datasets/nuswide_Query.txt", num_classes, 5000, discriminator_args)
            mAP_value = mAP.item()  # convert tensor to a Python number
            if mAP_value > MAX_mAP:
                MAX_mAP = mAP_value
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': generator.state_dict(),
                'best_mAP': MAX_mAP,
                'optimizer': g_optimizer.state_dict(),
                }, True, generator_args.DHD_output_dir, filename=job_name+".pth.tar")
                save_nets(job_name+"BEST_MAP", generator, epoch)
                print("SAVE THE BEST CHECKPOINT of DHD")
            print("mAP: ", mAP_value, "MAX mAP: ", MAX_mAP)

        # Save evaluation results in every epoch
        evaluation_metrics.append({
            'epoch': epoch,
            'valid_loss': valid_loss.item(),  
            'PSNR': valid_aux['psnr'], 
            'SSIM': valid_aux['ssim'],
            'LPIPS': valid_aux['lpips'],
            'DHD_MAXmap': MAX_mAP,
            'mAP': mAP_value,
            "HD_loss": HD_loss.item(),
            "val_hash": val_hash.item(),
            "val_hamming": val_hamming.item()
        })

        # Write results to JSON file
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

    # Save final epoch checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': generator.state_dict(),
        'best_mAP': MAX_mAP,
        'optimizer': g_optimizer.state_dict(),
        }, True, generator_args.DHD_output_dir, filename=job_name+"_last_epoch.pth.tar")
    print("SAVE FINAL EPOCH CHECKPOINT of DHD")

def validate_epoch(args, loader, model, lpips_model):
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
                lpips_val = lpips_model(images, output).mean()

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())
                epoch_postfix['lpips'] = '{:.4f}'.format(lpips_val.item())

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
                lpips_hist.append(lpips_val.item())
            
            loss_mean = np.nanmean(loss_hist)

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

    return loss_mean, return_aux

if __name__ == '__main__':
    DHD_checkpoint = './models_DHD/checkpoint.pth.tar'
    args.JSCC_P1 = 10
    args.JSCC_P2 = 10
    args1 = args

    train_DHD = True
    job_name = 'TESTTEST_Stage2_TRAINJSCC_with_Lpips_{P_value}dB_SNR_0709_valDH.pth'
    evaluate_JSCC_path1 = None
    train_gan(args, args1, job_name, evaluate_JSCC_path1, DHD_checkpoint)
