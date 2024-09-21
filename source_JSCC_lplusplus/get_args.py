import argparse

# def get_args():
#     ################################
#     # Setup Parameters and get args
#     ################################
#     parser = argparse.ArgumentParser()

#     parser.add_argument('-dataset', default  = 'cifar')

#     # The ViT setting
#     parser.add_argument('-n_patches', type=int, default  = 64)
#     parser.add_argument('-n_feat', type=int, default  = 48)
#     parser.add_argument('-hidden_size', type=int, default  = 256)
#     parser.add_argument('-feedforward_size', type=int, default  = 1024)
#     parser.add_argument('-n_layers', type=int, default  = 8)
#     parser.add_argument('-dropout_prob', type=float, default  = 0.1)
    
#     # The Swin setting
#     parser.add_argument('-image_dims', default  = [32, 32])
#     parser.add_argument('-depth', default  = [2, 4])
#     parser.add_argument('-embed_size', type=int, default  = 256)
#     parser.add_argument('-window_size', type=int, default  = 8)
#     parser.add_argument('-mlp_ratio', type=float, default  = 4)

#     parser.add_argument('-n_trans_feat', type=int, default  = 16)
#     parser.add_argument('-n_heads', type=int, default  = 8)

#     # The bandwidth adaption setting -> 2 types; one using different feats, one using different patches
#     parser.add_argument('-min_trans_feat', type=int, default  = 1)
#     parser.add_argument('-max_trans_feat', type=int, default  = 6)
#     parser.add_argument('-unit_trans_feat', type=int, default  = 4)
#     parser.add_argument('-trg_trans_feat', type=int, default  = 6)       # should be consistent with args.n_trans_feat
    

#     parser.add_argument('-min_trans_patch', type=int, default  = 5)
#     parser.add_argument('-max_trans_patch', type=int, default  = 8)
#     parser.add_argument('-unit_trans_patch', type=int, default  = 8)
#     parser.add_argument('-trg_trans_patch', type=int, default  = 5)       # should be consistent with args.n_trans_feat

#     parser.add_argument('-n_adapt_embed', type=int, default  = 2)
    
#     # channel
#     parser.add_argument('-channel_mode', default = 'awgn')
#     parser.add_argument('-link_qual',  default  = 7.0)
#     parser.add_argument('-link_rng',  default  = 3.0)


#     parser.add_argument('-adapt', default  = True)
#     parser.add_argument('-full_adapt', default  = True)

#     # dynamic weight adaption -- initial at 1; maximum 10; 
#     parser.add_argument('-threshold', default  = 0.25)            # if it is smaller than 0.25 dB, then it's fine
#     parser.add_argument('-min_clip', default  = 0)               # no smaller than 0
#     parser.add_argument('-max_clip', default  = 10)              # no larger than 10
#     parser.add_argument('-alpha', default  = 2)                  # weight[l] = 2**(alpha*delta[l])-1
#     parser.add_argument('-freq', default  = 1)                   # The frequency of updating the weights

#     # training setting
#     parser.add_argument('-epoch', type=int, default  = 4000)
#     parser.add_argument('-lr', type=float, default  = 1e-4)
#     parser.add_argument('-train_patience', type=int, default  = 80)
#     parser.add_argument('-train_batch_size', type=int, default  = 32)

#     parser.add_argument('-val_batch_size', type=int, default  = 32)
#     parser.add_argument('-resume', default  = False)
#     parser.add_argument('-path', default  = 'models/')

#     args = parser.parse_args()

#     return args

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()
    ### Arguments for DHD (Discriminator)
    parser.add_argument('--DHD_gpu_id', default="0", type=str, help="""Define GPU id.""")
    parser.add_argument('--DHD_data_dir', default="/datasets", type=str, help="""Path to dataset.""")
    parser.add_argument('--DHD_dataset', default="nuswide", type=str, help="""Dataset name: imagenet, nuswide_m, coco.""")
    
    parser.add_argument('--DHD_batch_size', default=16, type=int, help="""Training mini-batch size.""")
    # parser.add_argument('--DHD_num_workers', default=12, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--DHD_num_workers', default=2, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--DHD_encoder', default="AlexNet", type=str, help="""Encoder network: ResNet, AlexNet, ViT, DeiT, SwinT.""")
    parser.add_argument('--DHD_N_bits', default=64, type=int, help="""Number of bits to retrieval.""")
    parser.add_argument('--DHD_init_lr', default=3e-4, type=float, help="""Initial learning rate.""")
    parser.add_argument('--DHD_warm_up', default=10, type=int, help="""Learning rate warm-up end.""")
    parser.add_argument('--DHD_lambda1', default=0.1, type=float, help="""Balancing hyper-paramter on self knowledge distillation.""")
    parser.add_argument('--DHD_lambda2', default=0.1, type=float, help="""Balancing hyper-paramter on bce quantization.""")
    parser.add_argument('--DHD_std', default=0.5, type=float, help="""Gaussian estimator standrad deviation.""")
    parser.add_argument('--DHD_temp', default=0.2, type=float, help="""Temperature scaling parameter on hash proxy loss.""")
    parser.add_argument('--DHD_transformation_scale', default=0.2, type=float, help="""Transformation scaling for self teacher: AlexNet=0.2, else=0.5.""")

    parser.add_argument('--DHD_max_epoch', default=500, type=int, help="""Number of epochs to train.""")
    parser.add_argument('--DHD_eval_epoch', default=1, type=int, help="""Compute mAP for Every N-th epoch.""")
    parser.add_argument('--DHD_eval_init', default=1, type=int, help="""Compute mAP after N-th epoch.""")
    parser.add_argument('--DHD_output_dir', default=".", type=str, help="""Path to save logs and checkpoints.""")

    parser.add_argument('-dataset', default  = 'cifar')

    # The ViT setting
    parser.add_argument('-n_patches', type=int, default  = 64)
    parser.add_argument('-n_feat', type=int, default  = 48)
    parser.add_argument('-hidden_size', type=int, default  = 256)
    parser.add_argument('-feedforward_size', type=int, default  = 1024)
    parser.add_argument('-n_layers', type=int, default  = 8)
    parser.add_argument('-dropout_prob', type=float, default  = 0.1)
    
    # The Swin setting
    parser.add_argument('-image_dims', default  = [32, 32])
    parser.add_argument('-depth', default  = [2, 4])
    parser.add_argument('-embed_size', type=int, default  = 256)
    parser.add_argument('-window_size', type=int, default  = 8)
    parser.add_argument('-mlp_ratio', type=float, default  = 4)

    parser.add_argument('-n_trans_feat', type=int, default  = 16)
    parser.add_argument('-n_heads', type=int, default  = 8)

    # The bandwidth adaption setting -> 2 types; one using different feats, one using different patches
    parser.add_argument('-min_trans_feat', type=int, default  = 6)
    parser.add_argument('-max_trans_feat', type=int, default  = 6)
    parser.add_argument('-unit_trans_feat', type=int, default  = 4)
    parser.add_argument('-trg_trans_feat', type=int, default  = 6)       # should be consistent with args.n_trans_feat
    

    parser.add_argument('-min_trans_patch', type=int, default  = 5)
    parser.add_argument('-max_trans_patch', type=int, default  = 8)
    parser.add_argument('-unit_trans_patch', type=int, default  = 8)
    parser.add_argument('-trg_trans_patch', type=int, default  = 5)       # should be consistent with args.n_trans_feat

    parser.add_argument('-n_adapt_embed', type=int, default  = 2)
    
    # channel
    parser.add_argument('-channel_mode', default = 'awgn')
    parser.add_argument('-link_qual',  default  = 7.0)
    parser.add_argument('-link_rng',  default  = 3.0)


    parser.add_argument('-adapt', default  = True)
    parser.add_argument('-full_adapt', default  = True)

    # dynamic weight adaption -- initial at 1; maximum 10; 
    parser.add_argument('-threshold', default  = 0.25)            # if it is smaller than 0.25 dB, then it's fine
    parser.add_argument('-min_clip', default  = 0)               # no smaller than 0
    parser.add_argument('-max_clip', default  = 10)              # no larger than 10
    parser.add_argument('-alpha', default  = 2)                  # weight[l] = 2**(alpha*delta[l])-1
    parser.add_argument('-freq', default  = 1)                   # The frequency of updating the weights

    # training setting
    parser.add_argument('-epoch', type=int, default  = 4000)
    parser.add_argument('-lr', type=float, default  = 1e-4)
    parser.add_argument('-train_patience', type=int, default  = 80)
    parser.add_argument('-train_batch_size', type=int, default  = 128)

    parser.add_argument('-val_batch_size', type=int, default  = 64)
    parser.add_argument('-resume', default  = False)
    parser.add_argument('-path', default  = 'models/')

    args = parser.parse_args()

    return args
