import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
import lpips  # 导入lpips库

from myutils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models2 import *

parser = argparse.ArgumentParser()
model = 'dehazeformer-t'
num_workers=8
data_dir='./data/'
sava_dir='./saved_models/'
result_dir='./results/'
dataset='Haze1k_moderate'
exp='indoor'
args = parser.parse_args()

num = 0
PSNRsum = 0.0
SSIMsum = 0.0
LPIPSsum = 0.0  # 新增LPIPS总和变量

# 设置设备（CPU或GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# LPIPS计算器
loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

def single(save_dir):
    state_dict = torch.load(save_dir, map_location=device)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # 过滤掉包含 "total_ops" 和 "total_params" 的键
        if "total_ops" in k or "total_params" in k:
            continue
        name = k[7:] if k.startswith("module.") else k  # 移除 'module.' 前缀
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    global num, PSNRsum, SSIMsum, LPIPSsum
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    LPIPS_meter = AverageMeter()  # 用于存储LPIPS的平均值
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        num += 1
        input = batch['source'].to(device)
        target = batch['target'].to(device)

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            # 计算 PSNR
            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            # 计算 SSIM
            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(
                F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                data_range=1, size_average=False).item()

            # 计算 LPIPS
            lpips_val = loss_fn_lpips(output, target).item()

            PSNR.update(psnr_val)
            SSIM.update(ssim_val)
            LPIPS_meter.update(lpips_val)  # 更新LPIPS
            print('Test: [{0}]\t'
                  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
                  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
                  'LPIPS: {lpips_val:.03f} ({lpips_avg:.03f})'
                  .format(idx, psnr=PSNR, ssim=SSIM, lpips_val=lpips_val, lpips_avg=LPIPS_meter.avg))

            PSNRsum += psnr_val
            SSIMsum += ssim_val
            LPIPSsum += lpips_val  # 记录LPIPS总和
            f_result.write('%s,%.02f,%.03f,%.03f\n' % (filename, psnr_val, ssim_val, lpips_val))

            out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
            write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    PSNRaver = PSNRsum / num
    SSIMaver = SSIMsum / num
    LPIPSaver = LPIPSsum / num  # 计算LPIPS平均值
    print('PSNR值为{:.02f}'.format(PSNRaver))
    print('SSIM值为{:.03f}'.format(SSIMaver))
    print('LPIPS值为{:.03f}'.format(LPIPSaver))  # 打印LPIPS平均值
    f_result.close()

    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, '{:.02f} | {:.04f} | {:.03f}.csv'.format(PSNRaver, SSIMaver, LPIPSaver)))


def infer_only(input_dir, network, result_dir):
    """仅根据hazy目录做前向推理，不计算成对指标。"""
    import glob
    from PIL import Image
    import numpy as np

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    network.eval()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    image_paths = sorted(glob.glob(os.path.join(input_dir, '*')))  # e.g. (1).png
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        try:
            with torch.no_grad():
                img = Image.open(img_path).convert('RGB')
                arr = np.array(img).astype('float32') / 255.0  # [0,1]
                chw = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                # 映射到[-1,1]以匹配训练/评估流程
                chw = chw * 2.0 - 1.0
                output = network(chw).clamp_(-1, 1)
                output = output * 0.5 + 0.5  # 回到[0,1]
                out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
                write_img(os.path.join(result_dir, 'imgs', filename), out_img)
                print(f"Saved dehazed image: {os.path.join(result_dir, 'imgs', filename)}")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")


if __name__ == '__main__':
    network = eval(model.replace('-', '_'))()
    network.to(device)
    saved_model_dir = os.path.join(model + '.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + model)
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(data_dir, dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test') if os.path.isdir(os.path.join(dataset_dir, 'test', 'clear')) else None

    if test_dataset is None:
        hazy_dir = os.path.join(dataset_dir, 'test', 'hazy')
        result_dir = os.path.join(result_dir, dataset, model)
        infer_only(hazy_dir, network, result_dir)
    else:
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=num_workers,
                                 pin_memory=True)
        result_dir = os.path.join(result_dir, dataset, model)
        test(test_loader, network, result_dir)
