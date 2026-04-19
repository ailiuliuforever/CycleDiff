#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE 模型评估脚本
功能：
1. 定性评估：保存原始图像和重建图像的对比
2. 定量评估：计算 MSE、PSNR、SSIM、MS-SSIM、LPIPS、KL 散度等指标
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import yaml
import argparse
import math
from pathlib import Path
import torchvision as tv
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name, safe_torch_load
from util.mse_psnr_ssim_mssim import calculate_mse, calculate_psnr, calculate_ssim, calculate_msssim
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAE model")
    parser.add_argument("--cfg", type=str, 
                        default="configs/rsi2map/map_ae_kl_256x256_d4.yaml",
                        help="Config file path")
    parser.add_argument("--ckpt", type=str, 
                        default="results/map_ae_kl_256x256_d4/model-10.pt",
                        help="VAE checkpoint path")
    parser.add_argument("--save_dir", type=str, 
                        default="evaluation/vae/res",
                        help="Results save directory")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for evaluation")
    parser.add_argument("--cal_metrics", action="store_true", 
                        help="Calculate quantitative metrics")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--use_test_set", action="store_true",
                        help="Use test set instead of train set")
    args = parser.parse_args()
    return args


def load_conf(config_file):
    """加载配置文件"""
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf


def create_comparison_grid(original, reconstructed, num_images=8):
    """创建原始图像和重建图像的对比网格"""
    # 取前 num_images 个样本
    original = original[:num_images]
    reconstructed = reconstructed[:num_images]
    
    # 如果尺寸不匹配，将重建图像 resize 到原始图像尺寸
    if original.shape[-2:] != reconstructed.shape[-2:]:
        reconstructed = torch.nn.functional.interpolate(
            reconstructed, 
            size=original.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
    
    # 创建交替排列的网格
    comparison = []
    for i in range(len(original)):
        comparison.append(original[i])
        comparison.append(reconstructed[i])
    
    comparison = torch.stack(comparison, dim=0)
    grid = tv.utils.make_grid(
        comparison, 
        nrow=2,  # 每行 2 张（原始 + 重建）
        normalize=True, 
        value_range=(-1, 1),
        padding=2,
        pad_value=1.0
    )
    return grid


def evaluate_reconstruction(model, dataloader, save_dir, device, num_samples=50):
    """
    定性评估：保存原始图像和重建图像的对比
    
    Args:
        model: VAE 模型
        dataloader: 数据加载器
        save_dir: 保存目录
        device: 设备
        num_samples: 评估样本数量
    """
    model.eval()
    
    os.makedirs(os.path.join(save_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "reconstructed"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "comparison"), exist_ok=True)
    
    total_samples = 0
    comparison_grids = []
    
    print("正在评估重建质量...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if total_samples >= num_samples:
                break
            
            img = batch['image'].to(device)
            batch_size = img.shape[0]
            
            # 前向传播：编码 - 解码
            xrec, qloss = model(img)
            
            # 检查并修复尺寸不匹配问题
            if img.shape[-2:] != xrec.shape[-2:]:
                xrec = torch.nn.functional.interpolate(
                    xrec, 
                    size=img.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 保存单个图像
            for i in range(min(batch_size, num_samples - total_samples)):
                # 保存原始图像
                original = tv.utils.make_grid(img[i:i+1], nrow=1, normalize=True, value_range=(-1, 1))
                tv.utils.save_image(original, 
                                  os.path.join(save_dir, "original", f"sample_{total_samples + i:03d}.png"))
                
                # 保存重建图像
                reconstructed = tv.utils.make_grid(xrec[i:i+1], nrow=1, normalize=True, value_range=(-1, 1))
                tv.utils.save_image(reconstructed, 
                                  os.path.join(save_dir, "reconstructed", f"sample_{total_samples + i:03d}.png"))
            
            # 创建对比网格
            grid = create_comparison_grid(img, xrec, num_images=min(batch_size, 8))
            comparison_grids.append(grid)
            
            total_samples += batch_size
            if batch_idx % 10 == 0:
                print(f"  已处理 {min(total_samples, num_samples)}/{num_samples} 个样本")
    
    # 保存对比网格
    for idx, grid in enumerate(comparison_grids[:10]):  # 保存前 10 个 batch 的对比
        tv.utils.save_image(grid, 
                          os.path.join(save_dir, "comparison", f"comparison_batch_{idx:03d}.png"))
    
    # 保存所有对比的综合图
    if comparison_grids:
        all_grids = torch.cat(comparison_grids[:5], dim=1)  # 合并前 5 个 batch
        tv.utils.save_image(all_grids, 
                          os.path.join(save_dir, "comparison", "all_comparison.png"))
    
    print(f"✓ 重建图像已保存到：{save_dir}")
    print(f"  - 原始图像：{save_dir}/original/")
    print(f"  - 重建图像：{save_dir}/reconstructed/")
    print(f"  - 对比图像：{save_dir}/comparison/")
    
    return total_samples


def calculate_kl_divergence(model, dataloader, device, num_samples=50):
    """
    计算 KL 散度（评估潜在空间与标准正态分布的差距）
    
    KL 散度越小，说明潜在空间越接近标准正态分布
    """
    model.eval()
    kl_divs = []
    total_samples = 0
    
    print("正在计算 KL 散度...")
    
    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break
            
            img = batch['image'].to(device)
            
            # 编码得到后验分布
            posterior = model.encode(img)
            
            # KL 散度 = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_div = posterior.kl()
            kl_divs.append(kl_div.mean().item())
            
            total_samples += img.shape[0]
    
    mean_kl = np.mean(kl_divs)
    std_kl = np.std(kl_divs)
    
    return mean_kl, std_kl


def calculate_reconstruction_metrics(save_dir, num_samples=50):
    """
    计算重建指标：MSE, PSNR, SSIM, MS-SSIM
    """
    original_path = os.path.join(save_dir, "original")
    reconstructed_path = os.path.join(save_dir, "reconstructed")
    
    print("正在计算重建指标...")
    
    # MSE
    mse = calculate_mse(reconstructed_path, original_path)
    print(f"  MSE: {mse:.6f}")
    
    # PSNR
    psnr = calculate_psnr(reconstructed_path, original_path)
    print(f"  PSNR: {psnr:.2f} dB")
    
    # SSIM
    ssim = calculate_ssim(reconstructed_path, original_path)
    print(f"  SSIM: {ssim:.4f}")
    
    # MS-SSIM
    msssim = calculate_msssim(reconstructed_path, original_path)
    print(f"  MS-SSIM: {msssim:.4f}")
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'ms_ssim': msssim
    }


def calculate_lpips(model, dataloader, device, num_samples=50):
    """
    计算 LPIPS（Learned Perceptual Image Patch Similarity）
    基于深度特征的感知相似度指标
    """
    from util.loss import LPIPS
    
    model.eval()
    lpips_model = LPIPS().eval().to(device)
    lpips_scores = []
    total_samples = 0
    
    print("正在计算 LPIPS...")
    
    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break
            
            img = batch['image'].to(device)
            xrec, _ = model(img)
            
            # LPIPS 计算
            lpips_score = lpips_model(img, xrec)
            lpips_scores.append(lpips_score.mean().item())
            
            total_samples += img.shape[0]
    
    mean_lpips = np.mean(lpips_scores)
    std_lpips = np.std(lpips_scores)
    
    return mean_lpips, std_lpips


def save_metrics(metrics, save_dir):
    """保存评估指标到文件"""
    metrics_path = os.path.join(save_dir, "evaluation_metrics.txt")
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("VAE 模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("定性评估:\n")
        f.write(f"  - 原始图像：{save_dir}/original/\n")
        f.write(f"  - 重建图像：{save_dir}/reconstructed/\n")
        f.write(f"  - 对比图像：{save_dir}/comparison/\n\n")
        
        f.write("定量评估:\n")
        for key, value in metrics.items():
            if isinstance(value, dict):
                f.write(f"  {key}:\n")
                for k, v in value.items():
                    f.write(f"    {k}: {v:.6f}\n")
            else:
                f.write(f"  {key}: {value:.6f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("评估完成!\n")
    
    print(f"✓ 评估报告已保存到：{metrics_path}")


def main(args):
    """主函数"""
    # 加载配置
    print("=" * 50)
    print("VAE 模型评估")
    print("=" * 50)
    
    print(f"\n1. 加载配置文件：{args.cfg}")
    cfg = load_conf(args.cfg)
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    
    # 设置测试集
    if args.use_test_set:
        data_cfg['split'] = 'test'
        print(f"   使用测试集进行评估")
    else:
        data_cfg['split'] = 'train'
        print(f"   使用训练集进行评估")
    
    # 加载模型
    print(f"\n2. 加载 VAE 模型：{args.ckpt}")
    model = construct_class_by_name(**model_cfg)
    
    # 加载预训练权重
    if os.path.exists(args.ckpt):
        print(f"   从 {args.ckpt} 加载权重...")
        model.init_from_ckpt(args.ckpt)
    else:
        print(f"   警告：权重文件不存在 {args.ckpt}，使用随机初始化")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 加载数据
    print(f"\n3. 加载数据集：{data_cfg['data_root']}")
    dataset = construct_class_by_name(**data_cfg)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    print(f"   数据集大小：{len(dataset)}")
    print(f"   Batch size: {args.batch_size}")
    
    # 创建保存目录
    print(f"\n4. 创建结果保存目录：{args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. 定性评估
    print("\n" + "=" * 50)
    print("定性评估")
    print("=" * 50)
    num_evaluated = evaluate_reconstruction(model, dataloader, args.save_dir, device, args.num_samples)
    
    # 2. 定量评估
    metrics = {}
    
    if args.cal_metrics:
        print("\n" + "=" * 50)
        print("定量评估")
        print("=" * 50)
        
        # KL 散度
        kl_mean, kl_std = calculate_kl_divergence(model, dataloader, device, args.num_samples)
        print(f"  KL Divergence: {kl_mean:.6f} ± {kl_std:.6f}")
        metrics['kl_divergence'] = {'mean': kl_mean, 'std': kl_std}
        
        # 重建指标
        print()
        recon_metrics = calculate_reconstruction_metrics(args.save_dir, args.num_samples)
        metrics.update(recon_metrics)
        
        # LPIPS
        try:
            lpips_mean, lpips_std = calculate_lpips(model, dataloader, device, args.num_samples)
            print(f"  LPIPS: {lpips_mean:.6f} ± {lpips_std:.6f}")
            metrics['lpips'] = {'mean': lpips_mean, 'std': lpips_std}
        except Exception as e:
            print(f"  LPIPS 计算失败：{e}")
    
    # 保存指标
    save_metrics(metrics, args.save_dir)
    
    # 评估总结
    print("\n" + "=" * 50)
    print("评估总结")
    print("=" * 50)
    
    if 'psnr' in metrics:
        psnr = metrics['psnr']
        if psnr > 30:
            print(f"✓ VAE 质量：优秀 (PSNR > 30 dB)")
        elif psnr > 25:
            print(f"✓ VAE 质量：良好 (PSNR = {psnr:.2f} dB)")
        else:
            print(f"⚠ VAE 质量：需改进 (PSNR = {psnr:.2f} dB)")
    
    if 'ssim' in metrics:
        ssim = metrics['ssim']
        if ssim > 0.95:
            print(f"✓ 结构相似性：优秀 (SSIM > 0.95)")
        elif ssim > 0.90:
            print(f"✓ 结构相似性：良好 (SSIM = {ssim:.4f})")
        else:
            print(f"⚠ 结构相似性：需改进 (SSIM = {ssim:.4f})")
    
    if 'kl_divergence' in metrics:
        kl = metrics['kl_divergence']['mean']
        if kl < 0.01:
            print(f"✓ 潜在空间分布：优秀 (KL < 0.01)")
        elif kl < 0.1:
            print(f"✓ 潜在空间分布：良好 (KL = {kl:.6f})")
        else:
            print(f"⚠ 潜在空间分布：需改进 (KL = {kl:.6f})")
    
    print("\n✓ 评估完成！")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
