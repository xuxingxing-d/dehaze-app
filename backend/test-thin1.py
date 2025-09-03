import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
import lpips  # 导入lpips库

from myutils import write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *

parser = argparse.ArgumentParser()
model = 'dehazeformer-t'  # 第1个网络
num_workers=8
data_dir='./data/'
sava_dir='./saved_models/'
result_dir='./results/'
dataset='Haze1k_thin'  # thin数据集
exp='indoor'
args = parser.parse_args()

# 设置设备（CPU或GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def infer_only(input_dir, network, result_dir):
    """仅根据hazy目录做前向推理，不计算成对指标。"""
    import glob
    from PIL import Image
    import numpy as np
    import shutil

    # 清理结果目录，防止不同用户图片混合
    imgs_dir = os.path.join(result_dir, 'imgs')
    if os.path.exists(imgs_dir):
        shutil.rmtree(imgs_dir)
        print(f"已清理旧的结果目录: {imgs_dir}")
    
    os.makedirs(imgs_dir, exist_ok=True)
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
        except Exception as e:
            pass  # 静默处理错误


if __name__ == '__main__':
    network = eval(model.replace('-', '_'))()
    network.to(device)
    saved_model_dir = os.path.join('thin1.pth')  # thin数据集 + 第1个网络的模型

    if os.path.exists(saved_model_dir):
        network.load_state_dict(single(saved_model_dir))
    else:
        exit(0)

    # 只进行去雾推理，不计算PSNR和SSIM
    dataset_dir = os.path.join(data_dir, dataset)
    hazy_dir = os.path.join(dataset_dir, 'test', 'hazy')
    result_dir = os.path.join(result_dir, dataset, model)
    infer_only(hazy_dir, network, result_dir)
