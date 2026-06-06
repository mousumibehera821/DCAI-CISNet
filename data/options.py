import argparse

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=5, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=_str2bool, default=True)
    parser.add_argument('--shuffle', type=_str2bool, default=True)
    parser.add_argument('--threads', type=int, default=0, help='number of threads for dataloader to use')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=_str2bool, default=False)
    parser.add_argument('--cos_restart', type=_str2bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=_str2bool, default=True, help='turn False to train without warmup') 

    # train datasets
    parser.add_argument('--data_train_lol_v1'       , type=str, default='/home/nitr/Desktop/Mousumi_Behera/Datasets/LOLv1/train')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='/home/nitr/Desktop/Mousumi_Behera/Datasets/LOL-v2/Real/train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='/home/nitr/Desktop/Mousumi_Behera/Datasets/LOL-v2/Synthetic/train')

    # validation input
    parser.add_argument('--data_val_lol_v1'         , type=str, default='/home/nitr/Desktop/Mousumi_Behera/Datasets/LOLv1/test/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='/home/nitr/Desktop/Mousumi_Behera/Datasets/LOL-v2/Real/test/low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='/home/nitr/Desktop/Mousumi_Behera/Datasets/LOL-v2/Synthetic/test/low')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='/home/nitr/Desktop/Mousumi_Behera/Datasets/LOLv1/test/high')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='/home/nitr/Desktop/Mousumi_Behera/Datasets/LOL-v2/Real/test/high')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='/home/nitr/Desktop/Mousumi_Behera/Datasets/LOL-v2/Synthetic/test/high')
    
    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights
    parser.add_argument('--DCAI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)
    
    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=_str2bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=_str2bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=_str2bool, default=True, help='if gradient fluctuates too much, turn-on it')
    
    
    # choose which dataset you want to train
    parser.add_argument('--dataset', type=str, default='lol_v1',
    choices=['lol_v1',
             'lolv2_real',
             'lolv2_syn'],
    help='Select the dataset to train on (default: %(default)s)')

    return parser
