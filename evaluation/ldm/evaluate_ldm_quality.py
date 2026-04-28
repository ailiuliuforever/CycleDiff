#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LDM 重建质量详细评估脚本
功能：
1. 检查 latent 值的统计分布
2. 评估从 C 列表重建的质量
3. 分析 perceptual loss

采样流程说明：
- 参考 train_ldm_swanlab2.py 中训练过程的采样流程（第321-342行）
- C列表重建：从噪声开始，UNet预测noise，用预提取的C替换UNet预测的C，逐步去噪

注：图像分量（C）本身无法通过RGB图像可视化，因此只评估从C列表重建图像的能力
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import yaml
import argparse
import torchvision as tv
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name, safe_torch_load, unnormalize_to_zero_to_one
import numpy as np
from fvcore.common.config import CfgNode


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LDM reconstruction quality")
    parser.add_argument("--cfg", type=str,
                        default="configs/maps/map_ddm_const4_ldm_unet6_114_ode_2.yaml",
                        help="LDM config file path")
    parser.add_argument("--ckpt", type=str,
                        default="results/maps/ddm_const_uncond_unet_ldm_map/model-10.pt",
                        help="LDM checkpoint path")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    return args


def load_conf(config_file):
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf


def load_ldm_model(cfg, ckpt_path, device):
    model_cfg = cfg['model']

    first_stage_cfg = model_cfg['first_stage']
    first_stage_model = construct_class_by_name(**first_stage_cfg)

    unet_cfg = model_cfg['unet']
    unet = construct_class_by_name(**unet_cfg)

    model_kwargs = {'model': unet, 'auto_encoder': first_stage_model, 'cfg': model_cfg}
    model_kwargs.update(model_cfg)
    model = construct_class_by_name(**model_kwargs)
    model_kwargs.pop('model')
    model_kwargs.pop('auto_encoder')

    if os.path.exists(ckpt_path):
        print(f"   从 {ckpt_path} 加载权重...")
        data = safe_torch_load(ckpt_path, map_location="cpu")
        if 'model' in data:
            model.load_state_dict(data['model'])
            if 'scale_factor' in data['model']:
                model.scale_factor = data['model']['scale_factor'].to(device)
        else:
            model.load_state_dict(data)
        print(f"   scale_factor: {model.scale_factor}")
    else:
        print(f"   警告：权重文件不存在 {ckpt_path}")
        return None

    model = model.to(device)
    return model


def sample_from_c_list_correct(model, batch_size, c_list, device):
    """
    使用与训练脚本 sample_fn_d 相同的采样流程，但用预提取的 C 替换 UNet 预测的 C。
    
    参考 train_ldm_swanlab2.py 第321-342行的采样流程：
    1. 从噪声开始（与 sample_fn_d 一致）
    2. UNet 预测 C 和 noise
    3. 用预提取的 C 替换 UNet 预测的 C
    4. 使用 UNet 预测的 noise
    5. 逐步去噪重建图像
    """
    sampling_timesteps = model.sampling_timesteps
    step = 1. / sampling_timesteps
    rho = 1.
    
    step_indices = torch.arange(sampling_timesteps, dtype=torch.float64, device=device)
    t_steps = (model.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
            step - model.sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    image_size = model.image_size
    channels = model.channels
    down_ratio = model.first_stage_model.down_ratio
    shape = (batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio)
    
    # 从噪声开始，与 sample_fn_d 一致
    x_next = torch.randn(shape, device=device, dtype=torch.float64) * t_steps[0]
    
    # c_list 是从 reverse_q_sample_c_list_concat 获取的
    # 需要反转以匹配 t_steps 的顺序（从小到大）
    c_list_reversed = list(reversed(c_list))
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        pred = model.model(x_cur, t_cur)
        _, noise = pred[:2]
        
        # 用预提取的 C 替换 UNet 预测的 C
        if i < len(c_list_reversed):
            C = c_list_reversed[i].to(torch.float64)
        else:
            C = pred[0].to(torch.float64)
        
        noise = noise.to(torch.float64)
        x0 = x_cur - C * t_cur - noise * t_cur
        x_next = x0 + t_next * C + t_next * noise
    
    z = x_next
    
    # 反缩放（与 sample() 方法一致）
    if model.scale_by_std:
        z = 1. / model.scale_factor * z.detach()
    
    # VAE 解码（与 sample() 方法一致）
    x_rec = model.first_stage_model.decode(z.to(torch.float32))
    x_rec = unnormalize_to_zero_to_one(x_rec)
    x_rec = torch.clamp(x_rec, min=0., max=1.)
    
    return x_rec


def check_latent_statistics(model, dataloader, device, num_samples=50):
    """检查 latent 空间的统计分布"""
    model.eval()

    latent_values = []
    latent_means = []
    latent_stds = []
    c_values = []
    noise_values = []
    total_samples = 0

    print("\n" + "=" * 60)
    print("Latent 空间统计分析")
    print("=" * 60)

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break

            img = batch['image'].to(device)

            z = model.first_stage_model.encode(img)
            z = model.get_first_stage_encoding(z)
            z_scaled = model.scale_factor * z

            latent_values.append(z_scaled.cpu())
            latent_means.append(z_scaled.mean().item())
            latent_stds.append(z_scaled.std().item())

            c_list, noise = model.reverse_q_sample_c_list_concat(img)
            for c in c_list:
                c_values.append(c.cpu())
            noise_values.append(noise.cpu())

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

    print(f"\nLatent shape: {all_latents.shape}")

    if c_values:
        all_c = torch.cat(c_values, dim=0)
        print(f"\nC 分量统计（共 {len(c_values)} 个时间步）：")
        print(f"  均值：{all_c.mean().item():.6f}")
        print(f"  标准差：{all_c.std().item():.6f}")
        print(f"  最小值：{all_c.min().item():.6f}")
        print(f"  最大值：{all_c.max().item():.6f}")
        print(f"  绝对值均值：{all_c.abs().mean().item():.6f}")

    if noise_values:
        all_noise = torch.cat(noise_values, dim=0)
        print(f"\n噪声分量统计：")
        print(f"  均值：{all_noise.mean().item():.6f}")
        print(f"  标准差：{all_noise.std().item():.6f}")
        print(f"  最小值：{all_noise.min().item():.6f}")
        print(f"  最大值：{all_noise.max().item():.6f}")

    return {
        'mean': all_latents.mean().item(),
        'std': all_latents.std().item(),
        'min': all_latents.min().item(),
        'max': all_latents.max().item(),
        'abs_mean': all_latents.abs().mean().item(),
    }


def check_reconstruction_quality(model, dataloader, device, num_samples=50):
    """评估从 C 列表重建的质量"""
    model.eval()

    mse_c_reconstructed = []
    total_samples = 0

    print("\n" + "=" * 60)
    print("重建质量分析")
    print("=" * 60)
    print("  采样流程：从噪声开始 → UNet预测noise → 用预提取C替换 → 逐步去噪")

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break

            img = batch['image'].to(device)

            c_list, _ = model.reverse_q_sample_c_list_concat(img)

            # C 列表重建（使用修正后的采样函数）
            x_c_reconstructed = sample_from_c_list_correct(model, img.shape[0], c_list, device)
            x_c_reconstructed_norm = x_c_reconstructed * 2.0 - 1.0

            if img.shape[-2:] != x_c_reconstructed_norm.shape[-2:]:
                x_c_reconstructed_norm = torch.nn.functional.interpolate(
                    x_c_reconstructed_norm, size=img.shape[-2:], mode='bilinear', align_corners=False
                )

            mse_cr = ((img - x_c_reconstructed_norm) ** 2).mean().item()
            mse_c_reconstructed.append(mse_cr)

            total_samples += img.shape[0]

    print(f"\n重建误差统计（共 {total_samples} 个样本）：")
    print(f"  MSE 均值：{np.mean(mse_c_reconstructed):.6f}")
    print(f"  MSE 标准差：{np.std(mse_c_reconstructed):.6f}")

    psnr_cr = []
    for mse in mse_c_reconstructed:
        if mse > 0:
            psnr_cr.append(10 * np.log10(4.0 / mse))

    if psnr_cr:
        print(f"  PSNR 均值：{np.mean(psnr_cr):.2f} dB")

    return {
        'mse_mean': np.mean(mse_c_reconstructed),
        'psnr_mean': np.mean(psnr_cr) if psnr_cr else 0,
    }


def check_perceptual_loss(model, dataloader, device, num_samples=50):
    """分析 Perceptual Loss"""
    from taming.modules.losses.lpips import LPIPS

    model.eval()
    lpips_model = LPIPS().eval().to(device)

    lpips_c_reconstructed = []
    total_samples = 0

    print("\n" + "=" * 60)
    print("Perceptual Loss (LPIPS) 分析")
    print("=" * 60)
    print("  采样流程：从噪声开始 → UNet预测noise → 用预提取C替换 → 逐步去噪")

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= num_samples:
                break

            img = batch['image'].to(device)

            c_list, _ = model.reverse_q_sample_c_list_concat(img)

            # C 列表重建（使用修正后的采样函数）
            x_c_reconstructed = sample_from_c_list_correct(model, img.shape[0], c_list, device)
            x_c_reconstructed_norm = x_c_reconstructed * 2.0 - 1.0

            if img.shape[-2:] != x_c_reconstructed_norm.shape[-2:]:
                x_c_reconstructed_norm = torch.nn.functional.interpolate(
                    x_c_reconstructed_norm, size=img.shape[-2:], mode='bilinear', align_corners=False
                )

            lpips_cr = lpips_model(img, x_c_reconstructed_norm)
            lpips_c_reconstructed.append(lpips_cr.mean().item())

            total_samples += img.shape[0]

    print(f"\nLPIPS 统计（共 {total_samples} 个样本）：")
    print(f"  均值：{np.mean(lpips_c_reconstructed):.6f}")
    print(f"  标准差：{np.std(lpips_c_reconstructed):.6f}")

    return {
        'lpips_mean': np.mean(lpips_c_reconstructed),
        'lpips_std': np.std(lpips_c_reconstructed),
    }


def main(args):
    print("=" * 60)
    print("LDM 重建质量详细评估")
    print("=" * 60)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(f"\n1. 加载配置文件：{args.cfg}")
    cfg = load_conf(args.cfg)
    data_cfg = cfg['data']
    data_cfg['split'] = 'train'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n2. 加载 LDM 模型：{args.ckpt}")
    model = load_ldm_model(cfg, args.ckpt, device)
    if model is None:
        return
    model.eval()

    print(f"\n3. 加载数据集：{data_cfg['data_root']}")
    dataset = construct_class_by_name(**data_cfg)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"   数据集大小：{len(dataset)}")

    latent_stats = check_latent_statistics(model, dataloader, device, args.num_samples)

    recon_stats = check_reconstruction_quality(model, dataloader, device, args.num_samples)

    try:
        lpips_stats = check_perceptual_loss(model, dataloader, device, args.num_samples)
    except Exception as e:
        print(f"\nLPIPS 计算失败：{e}")
        lpips_stats = {}

    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)

    print(f"\nLatent 空间：")
    print(f"  均值：{latent_stats['mean']:.6f}")
    print(f"  标准差：{latent_stats['std']:.6f}")
    print(f"  绝对值均值：{latent_stats['abs_mean']:.6f}")

    print(f"\n重建质量：")
    print(f"  MSE：{recon_stats['mse_mean']:.6f}")
    print(f"  PSNR：{recon_stats['psnr_mean']:.2f} dB")

    if lpips_stats:
        print(f"\nPerceptual Loss：")
        print(f"  LPIPS：{lpips_stats['lpips_mean']:.6f}")

    print("\n" + "=" * 60)
    print("✓ 评估完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
