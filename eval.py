import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CISNet import CISNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(model, testing_data_loader, model_path, output_folder,norm_size=True,LOL=False,v2=False,unpaired=False,alpha=1.0,gamma=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            input = input.to(device)
            output = model(input**gamma) 
            
        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)  
            
        output = torch.clamp(output.to(device),0,1).to(device)
        if not norm_size:
            output = output[:, :, :h, :w]
        
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    eval_parser = argparse.ArgumentParser(description='Eval')
    eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
    eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
    eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
    eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')

    eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
    eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
    eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

    eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
    eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
    eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
    eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
    eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
    eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
    eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
    eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
    eval_parser.add_argument('--alpha', type=float, default=1.0)
    eval_parser.add_argument('--gamma', type=float, default=1.0)
    eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth')

    ep = eval_parser.parse_args()


    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("/home/nitr/Desktop/Mousumi_Behera/Datasets/LOLv1/test/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        if ep.perc:
            weight_path = '/home/nitr/Desktop/Mousumi_Behera/Code/DCAI-CISNet/weights/train/LOLv1.pth'
        else:
            weight_path = '/home/nitr/Desktop/Mousumi_Behera/Code/DCAI-CISNet/weights/train/LOLv1.pth'
          
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("/home/nitr/Desktop/Mousumi_Behera/Datasets/LOL-v2/Real/test/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        if ep.best_GT_mean:
            weight_path = '/home/nitr/Desktop/Mousumi_Behera/Code/DCAI-CISNet/weights/train/LOLv2R.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = '/home/nitr/Desktop/Mousumi_Behera/Code/DCAI-CISNet/weights/train/LOLv2R.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = '/home/nitr/Desktop/Mousumi_Behera/Code/DCAI-CISNet/weights/train/LOLv2R.pth'
            alpha = 0.82
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("/home/nitr/Desktop/Mousumi_Behera/Datasets/LOL-v2/Synthetic/test/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        if ep.perc:
            weight_path = '/home/nitr/Desktop/Mousumi_Behera/Code/DCAI-CISNet/weights/train/LOLv2S.pth'
        else:
            weight_path = '/home/nitr/Desktop/Mousumi_Behera/Code/DCAI-CISNet/weights/train/LOLv2S.pth'
            
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/home/nitr/Desktop/Mousumi_Behera/Datasets/DICM/data/low"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/home/nitr/Desktop/Mousumi_Behera/Datasets/LIME/data/low"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/home/nitr/Desktop/Mousumi_Behera/Datasets/MEF/data/low"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/home/nitr/Desktop/Mousumi_Behera/Datasets/NPE/data/low"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("/home/nitr/Desktop/Mousumi_Behera/Datasets/VV/data/low"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/VV/'
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights
        
    eval_net = CISNet().cuda()
    eval(eval_net, eval_data, weight_path, output_folder,norm_size=norm_size,LOL=ep.lol,v2=ep.lol_v2_real,unpaired=ep.unpaired,alpha=alpha,gamma=ep.gamma)

