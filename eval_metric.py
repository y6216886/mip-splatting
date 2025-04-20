import torch
import os
import numpy as np
from tqdm import tqdm
import imageio
from argparse import ArgumentParser
import torch
from kornia.metrics import ssim as dssim
import metrics
import wandb

from PIL import Image
from torchvision import transforms as T

import lpips
lpips_alex = lpips.LPIPS(net='alex') 

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str,
                        default='./output/brandenburg_gate/reconstruction/debug',
                        help='root directory of dataset')


    return parser.parse_args()




def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3) # dissimilarity in [0, 1]
    return dssim_


# if __name__ == "__main__":
#     args = get_opts()

def calculate_metrics(exp_path, iswandb=False):
        farther_dir = exp_path
        gt_path= farther_dir + '/gt'
        pred_path= farther_dir + '/pred'
        imgs, psnrs, ssims, lpips_alexs, lpips_vggs, maes, mses = [], [], [], [], [], [], []

        toTensor = T.ToTensor()
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        gt_list = os.listdir(gt_path)
        # pred_list = os.listdir(pred_path)
        for i in tqdm(range(len(gt_list))):
            image_gt_path = os.path.join(gt_path,  gt_list[i])
            image_pred_path = os.path.join(pred_path,  gt_list[i])

            img_pred = Image.open(image_pred_path).convert('RGB')
            w, h = img_pred.size
            img_pred = toTensor(img_pred) # (3, h, w)
            normalize_img_pre = normalize(img_pred).unsqueeze(0)
            img_pred_ = (img_pred.permute(1, 2, 0).numpy()*255).astype(np.uint8)
            imgs += [img_pred_]
            
            img_gt = Image.open(image_gt_path).convert('RGB')
            # w, h = img_pred.size()
            img_gt = toTensor(img_gt) # (3, h, w)
            normalize_img_gt = normalize(img_gt).unsqueeze(0)
            # img_gt_ = (img_gt.permute(1, 2, 0).numpy()*255).astype(np.uint8)
            # img_gts += [img_gt_]

            # rgbs = sample['rgbs']
            # img_gt = rgbs.view(h, w, 3)
            # breakpoint()
            psnrs += [psnr(img_gt.permute(1, 2, 0)[:,w//2:,:], img_pred.permute(1, 2, 0)[:,w//2:,:])]
            ssims += [ssim(img_gt[:,:,w//2:][None,...], img_pred[:, :, w//2:][None,...])]
            lpips_alexs += [lpips_alex(normalize_img_gt[...,w//2:], normalize_img_pre[...,w//2:])]
            mses += [((img_gt.permute(1, 2, 0)[:,w//2:,:] - img_pred.permute(1, 2, 0)[:,w//2:,:])**2).mean()]
        print("image_pre_path",image_pred_path, "gt_path", image_gt_path)
        mean_psnr = torch.mean(torch.stack(psnrs)).item()
        mean_ssim = torch.mean(torch.stack([x.mean() for x in ssims])).item()
        mean_lpips_alex =torch.mean(torch.stack(lpips_alexs)).item()
        mean_mse =torch.mean(torch.stack(mses)).item()
        with open(os.path.join(farther_dir, 'result.txt'), "a") as f:
            f.write(f'metrics : \n')
            f.write(f'Mean PSNR : {mean_psnr:.4f}\n')
            f.write(f'Mean SSIM : {mean_ssim:.4f}\n')
            f.write(f'Mean LIPIS_alex : {mean_lpips_alex:.4f}\n')
            f.write(f'Mean MSE : {mean_mse:.4f}\n')
        if iswandb: wandb.log({'Mean PSNR': mean_psnr, 'Mean SSIM': mean_ssim, 'Mean LIPIS_alex': mean_lpips_alex, 'Mean MSE': mean_mse})
        print('Mean PSNR',mean_psnr)
        print('Mean SSIM' ,mean_ssim)
        print('Mean LIPIS_alex',mean_lpips_alex)

if __name__=="__main__":
    # /root/young/code/stylegs/output/brandenburg_gate/reconstruction/debug1
    calculate_metrics("/root/young/code/mip-splatting/output/app-random-app-maskrcnn")