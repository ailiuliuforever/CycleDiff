#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE 重建质量详细评估脚本
功能：
1. 检查 latent 值的统计分布
2. 评估 VAE 重建质量
3. 分析 perceptual loss 的大小
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import yaml
import argparse
from pathlib import Path
import torchvision as tv
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAE reconstruction quality")
    parser.add_argument("--cfg", type=str, 
                        default="configs/rsi2map/map_ae_kl_256x256_d4.yaml",
                        help="Config file path")
    parser.add_argument("--ckpt", type=str, 
                        default="results/map_ae_kl_256x256_d4/model-10.pt",
                        help="VAE checkpoint path")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate")
    args = parser.parse_args()
    return args


def load_conf(config_file):
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf


def check_latent_statistics(model, dataloader, device, num_samples=50):
    """检查 latent 值的统计分布"""
    model.eval()
    
    latent_values = []
    latent_means = []
    latent_stds = []
    latent_mins = []
    latent_maxs = []
    total_samples = 0
    
    print("\n" + "=" * 60)
    print("Latent 空间统计分析")
    print("=" * 60)
    
    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break
            
            img = batch['image'].to(device)
            
            # 编码得到 latent
            posterior = model.encode(img)
            z = posterior.sample()
            
            latent_values.append(z.cpu())
            latent_means.append(z.mean().item())
            latent_stds.append(z.std().item())
            latent_mins.append(z.min().item())
            latent_maxs.append(z.max().item())
            
            total_samples += img.shape[0]
    
    all_latents = torch.cat(latent_values, dim=0)
    
    print(f"\nLatent 值统计（共 {total_samples} 个样本）：")
    print(f"  整体均值：{all_latents.mean().item():.6f}")
    print(f"  整体标准差：{all_latents.std().item():.6f}")
    print(f"  整体最小值：{all_latents.min().item():.6f}")
    print(f"  整体最大值：{all_latents.max().item():.6f}")
    print(f"  绝对值均值：{all_latents.abs().mean().item():.6f}")
    
    print(f"\n每个 batch 的均值范围：{min(latent_means):.6f} ~ {max(latent_means):.6f}")
    print(f"每个 batch 的标准差范围：{min(latent_stds):.6f} ~ {max(latent_stds):.6f}")
    print(f"每个 batch 的最小值范围：{min(latent_mins):.6f} ~ {max(latent_mins):.6f}")
    print(f"每个 batch 的最大值范围：{min(latent_maxs):.6f} ~ {max(latent_maxs):.6f}")
    
    print(f"\nLatent shape: {all_latents.shape}")
    print(f"  Batch size: {all_latents.shape[0]}")
    print(f"  Channels: {all_latents.shape[1]}")
    print(f"  Height: {all_latents.shape[2]}")
    print(f"  Width: {all_latents.shape[3]}")
    
    return {
        'mean': all_latents.mean().item(),
        'std': all_latents.std().item(),
        'min': all_latents.min().item(),
        'max': all_latents.max().item(),
        'abs_mean': all_latents.abs().mean().item(),
    }


def check_reconstruction_quality(model, dataloader, device, num_samples=50):
    """检查重建质量"""
    model.eval()
    
    mse_values = []
    mae_values = []
    total_samples = 0
    
    print("\n" + "=" * 60)
    print("重建质量分析")
    print("=" * 60)
    
    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break
            
            img = batch['image'].to(device)
            xrec, qloss = model(img)
            
            # 检查并修复尺寸不匹配
            if img.shape[-2:] != xrec.shape[-2:]:
                xrec = torch.nn.functional.interpolate(
                    xrec, size=img.shape[-2:], mode='bilinear', align_corners=False
                )
            
            # 计算 MSE 和 MAE
            mse = ((img - xrec) ** 2).mean().item()
            mae = (img - xrec).abs().mean().item()
            
            mse_values.append(mse)
            mae_values.append(mae)
            
            total_samples += img.shape[0]
    
    print(f"\n重建误差统计（共 {total_samples} 个样本）：")
    print(f"  MSE 均值：{np.mean(mse_values):.6f}")
    print(f"  MSE 标准差：{np.std(mse_values):.6f}")
    print(f"  MAE 均值：{np.mean(mae_values):.6f}")
    print(f"  MAE 标准差：{np.std(mae_values):.6f}")
    
    # PSNR
    psnr_values = []
    for mse in mse_values:
        if mse > 0:
            psnr = 10 * np.log10(4.0 / mse)  # range [-1, 1] -> max^2 = 4
            psnr_values.append(psnr)
    
    print(f"  PSNR 均值：{np.mean(psnr_values):.2f} dB")
    print(f"  PSNR 标准差：{np.std(psnr_values):.2f} dB")
    
    return {
        'mse_mean': np.mean(mse_values),
        'mae_mean': np.mean(mae_values),
        'psnr_mean': np.mean(psnr_values),
    }


def check_perceptual_loss(model, dataloader, device, num_samples=50):
    """检查 perceptual loss 大小"""
    from taming.modules.losses.lpips import LPIPS
    
    model.eval()
    lpips_model = LPIPS().eval().to(device)
    
    lpips_values = []
    total_samples = 0
    
    print("\n" + "=" * 60)
    print("Perceptual Loss (LPIPS) 分析")
    print("=" * 60)
    
    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break
            
            img = batch['image'].to(device)
            xrec, qloss = model(img)
            
            # 检查并修复尺寸不匹配
            if img.shape[-2:] != xrec.shape[-2:]:
                xrec = torch.nn.functional.interpolate(
                    xrec, size=img.shape[-2:], mode='bilinear', align_corners=False
                )
            
            # 计算 LPIPS
            lpips_score = lpips_model(img, xrec)
            lpips_values.append(lpips_score.mean().item())
            
            total_samples += img.shape[0]
    
    print(f"\nLPIPS 统计（共 {total_samples} 个样本）：")
    print(f"  LPIPS 均值：{np.mean(lpips_values):.6f}")
    print(f"  LPIPS 标准差：{np.std(lpips_values):.6f}")
    print(f"  LPIPS 最小值：{min(lpips_values):.6f}")
    print(f"  LPIPS 最大值：{max(lpips_values):.6f}")
    
    return {
        'lpips_mean': np.mean(lpips_values),
        'lpips_std': np.std(lpips_values),
    }


def main(args):
    print("=" * 60)
    print("VAE 重建质量详细评估")
    print("=" * 60)
    
    print(f"\n1. 加载配置文件：{args.cfg}")
    cfg = load_conf(args.cfg)
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    
    # 设置训练集
    data_cfg['split'] = 'train'
    
    print(f"\n2. 加载 VAE 模型：{args.ckpt}")
    model = construct_class_by_name(**model_cfg)
    
    if os.path.exists(args.ckpt):
        print(f"   从 {args.ckpt} 加载权重...")
        model.init_from_ckpt(args.ckpt)
    else:
        print(f"   警告：权重文件不存在 {args.ckpt}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\n3. 加载数据集：{data_cfg['data_root']}")
    dataset = construct_class_by_name(**data_cfg)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"   数据集大小：{len(dataset)}")
    
    # 1. 检查 latent 统计
    latent_stats = check_latent_statistics(model, dataloader, device, args.num_samples)
    
    # 2. 检查重建质量
    recon_stats = check_reconstruction_quality(model, dataloader, device, args.num_samples)
    
    # 3. 检查 perceptual loss
    lpips_stats = check_perceptual_loss(model, dataloader, device, args.num_samples)
    
    # 总结
    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)
    
    print(f"\nLatent 空间：")
    print(f"  均值：{latent_stats['mean']:.6f}")
    print(f"  标准差：{latent_stats['std']:.6f}")
    print(f"  绝对值均值：{latent_stats['abs_mean']:.6f}")
    
    print(f"\n重建质量：")
    print(f"  MSE：{recon_stats['mse_mean']:.6f}")
    print(f"  MAE：{recon_stats['mae_mean']:.6f}")
    print(f"  PSNR：{recon_stats['psnr_mean']:.2f} dB")
    
    print(f"\nPerceptual Loss：")
    print(f"  LPIPS：{lpips_stats['lpips_mean']:.6f}")
    
    print("\n" + "=" * 60)
    print("✓ 评估完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
