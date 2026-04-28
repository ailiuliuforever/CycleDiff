#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LDM 模型评估脚本
功能：
评估从图像分量（C列表）重建原始图像的能力

采样流程说明：
- 参考 train_ldm_swanlab2.py 中训练过程的采样流程（第321-342行）
- 从噪声开始，UNet预测noise，用预提取的C替换UNet预测的C，逐步去噪
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import yaml
import argparse
import copy
import torchvision as tv
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name, safe_torch_load, unnormalize_to_zero_to_one
from ddm.encoder_decoder import AutoencoderKL
from util.mse_psnr_ssim_mssim import calculate_mse, calculate_psnr, calculate_ssim, calculate_msssim
import numpy as np
from fvcore.common.config import CfgNode


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LDM model - Reconstruction from C list")
    parser.add_argument("--cfg", type=str,
                        default="configs/maps/map_ddm_const4_ldm_unet6_114_ode_2.yaml",
                        help="LDM config file path")
    parser.add_argument("--ckpt", type=str,
                        default="results/maps/ddm_const_uncond_unet_ldm_map/model-10.pt",
                        help="LDM checkpoint path")
    parser.add_argument("--save_dir", type=str,
                        default="evaluation/ldm/res",
                        help="Results save directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--cal_metrics", action="store_true",
                        help="Calculate quantitative metrics")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--use_test_set", action="store_true",
                        help="Use test set instead of train set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
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
        print(f"   警告：权重文件不存在 {ckpt_path}，使用随机初始化")

    model = model.to(device)
    model.eval()
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
    
    Args:
        model: LatentDiffusion 模型
        batch_size: batch size
        c_list: 预提取的 C 列表（从 reverse_q_sample_c_list_concat 获取）
        device: 设备
    
    Returns:
        x_rec: 重建图像 [0, 1] 归一化
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
            # 如果 C 列表不够长，使用 UNet 预测的 C
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


def create_dual_comparison_grid(original, reconstructed, num_images=4):
    """创建双对比图（原始 vs 重建）"""
    original = original[:num_images]
    reconstructed = reconstructed[:num_images]

    if original.shape[-2:] != reconstructed.shape[-2:]:
        reconstructed = torch.nn.functional.interpolate(
            reconstructed, size=original.shape[-2:], mode='bilinear', align_corners=False
        )

    comparison = []
    for i in range(len(original)):
        comparison.append(original[i])
        comparison.append(reconstructed[i])

    comparison = torch.stack(comparison, dim=0)
    grid = tv.utils.make_grid(
        comparison,
        nrow=2,
        normalize=True,
        value_range=(0, 1),
        padding=2,
        pad_value=1.0
    )
    return grid


def evaluate_reconstruction_from_c(model, dataloader, save_dir, device, num_samples=50):
    """评估从图像分量（C列表）重建原始图像的能力"""
    model.eval()

    os.makedirs(os.path.join(save_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "reconstructed"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "comparison"), exist_ok=True)

    total_samples = 0
    comparison_grids = []

    print("正在评估从图像分量重建原始图像的能力...")
    print("  采样流程：从噪声开始 → UNet预测noise → 用预提取C替换UNet预测的C → 逐步去噪")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if total_samples >= num_samples:
                break

            img = batch['image'].to(device)
            batch_size = img.shape[0]

            c_list, _ = model.reverse_q_sample_c_list_concat(img)

            # 使用修正后的采样函数
            x_rec = sample_from_c_list_correct(model, batch_size, c_list, device)

            img_display = (img + 1.0) / 2.0

            for i in range(min(batch_size, num_samples - total_samples)):
                orig = tv.utils.make_grid(img_display[i:i+1], nrow=1, normalize=False)
                tv.utils.save_image(orig,
                                  os.path.join(save_dir, "original", f"sample_{total_samples + i:03d}.png"))

                rec = tv.utils.make_grid(x_rec[i:i+1], nrow=1, normalize=False)
                tv.utils.save_image(rec,
                                  os.path.join(save_dir, "reconstructed", f"sample_{total_samples + i:03d}.png"))

            n_compare = min(batch_size, 4)
            grid = create_dual_comparison_grid(
                img_display[:n_compare],
                x_rec[:n_compare],
                num_images=n_compare
            )
            comparison_grids.append(grid)

            total_samples += batch_size
            if batch_idx % 5 == 0:
                print(f"  已处理 {min(total_samples, num_samples)}/{num_samples} 个样本")

    for idx, grid in enumerate(comparison_grids[:10]):
        tv.utils.save_image(grid,
                          os.path.join(save_dir, "comparison", f"comparison_batch_{idx:03d}.png"))

    if comparison_grids:
        all_grids = torch.cat(comparison_grids[:5], dim=1)
        tv.utils.save_image(all_grids,
                          os.path.join(save_dir, "comparison", "all_comparison.png"))

    print(f"✓ 重建结果已保存到：{save_dir}")
    print(f"  - 原始图像：{save_dir}/original/")
    print(f"  - 重建图像：{save_dir}/reconstructed/")
    print(f"  - 对比图像：{save_dir}/comparison/")

    return total_samples


def calculate_reconstruction_metrics(save_dir, num_samples=50):
    """计算重建指标"""
    original_path = os.path.join(save_dir, "original")
    reconstructed_path = os.path.join(save_dir, "reconstructed")

    print(f"正在计算重建指标...")

    mse = calculate_mse(reconstructed_path, original_path)
    print(f"  MSE: {mse:.6f}")

    psnr = calculate_psnr(reconstructed_path, original_path)
    print(f"  PSNR: {psnr:.2f} dB")

    ssim = calculate_ssim(reconstructed_path, original_path)
    print(f"  SSIM: {ssim:.4f}")

    msssim = calculate_msssim(reconstructed_path, original_path)
    print(f"  MS-SSIM: {msssim:.4f}")

    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'ms_ssim': msssim
    }


def calculate_lpips(model, dataloader, device, num_samples=50):
    """计算 LPIPS"""
    from taming.modules.losses.lpips import LPIPS

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

            c_list, _ = model.reverse_q_sample_c_list_concat(img)
            x_rec = sample_from_c_list_correct(model, img.shape[0], c_list, device)

            x_rec_norm = x_rec * 2.0 - 1.0

            lpips_score = lpips_model(img, x_rec_norm)
            lpips_scores.append(lpips_score.mean().item())

            total_samples += img.shape[0]

    mean_lpips = np.mean(lpips_scores)
    std_lpips = np.std(lpips_scores)

    return mean_lpips, std_lpips


def save_metrics(metrics, save_dir):
    """保存评估指标到文件"""
    metrics_path = os.path.join(save_dir, "evaluation_metrics.txt")

    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("LDM 模型评估报告 - 从图像分量重建原始图像\n")
        f.write("=" * 60 + "\n\n")

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

        f.write("\n" + "=" * 60 + "\n")
        f.write("评估完成!\n")

    print(f"✓ 评估报告已保存到：{metrics_path}")


def main(args):
    print("=" * 60)
    print("LDM 模型评估 - 从图像分量重建原始图像")
    print("=" * 60)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(f"\n1. 加载配置文件：{args.cfg}")
    cfg = load_conf(args.cfg)
    data_cfg = cfg['data']

    if args.use_test_set:
        data_cfg['split'] = 'test'
        print(f"   使用测试集进行评估")
    else:
        data_cfg['split'] = 'train'
        print(f"   使用训练集进行评估")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n2. 加载 LDM 模型：{args.ckpt}")
    model = load_ldm_model(cfg, args.ckpt, device)

    print(f"\n3. 加载数据集：{data_cfg['data_root']}")
    dataset = construct_class_by_name(**data_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print(f"   数据集大小：{len(dataset)}")
    print(f"   Batch size: {args.batch_size}")

    print(f"\n4. 创建结果保存目录：{args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("评估：从图像分量（C列表）重建原始图像的能力")
    print("=" * 60)
    num_evaluated = evaluate_reconstruction_from_c(model, dataloader, args.save_dir, device, args.num_samples)

    metrics = {}

    if args.cal_metrics:
        print("\n" + "=" * 60)
        print("定量评估")
        print("=" * 60)

        reconstruction_metrics = calculate_reconstruction_metrics(args.save_dir, args.num_samples)
        metrics['reconstruction'] = reconstruction_metrics

        try:
            lpips_mean, lpips_std = calculate_lpips(model, dataloader, device, args.num_samples)
            print(f"  LPIPS: {lpips_mean:.6f} ± {lpips_std:.6f}")
            metrics['lpips'] = {'mean': lpips_mean, 'std': lpips_std}
        except Exception as e:
            print(f"  LPIPS 计算失败：{e}")

    save_metrics(metrics, args.save_dir)

    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)

    if 'reconstruction' in metrics:
        rec = metrics['reconstruction']
        if 'psnr' in rec:
            psnr = rec['psnr']
            if psnr > 30:
                print(f"✓ 重建质量：优秀 (PSNR > 30 dB)")
            elif psnr > 25:
                print(f"✓ 重建质量：良好 (PSNR = {psnr:.2f} dB)")
            else:
                print(f"⚠ 重建质量：需改进 (PSNR = {psnr:.2f} dB)")
        if 'ssim' in rec:
            ssim = rec['ssim']
            if ssim > 0.95:
                print(f"✓ 结构相似性：优秀 (SSIM > 0.95)")
            elif ssim > 0.90:
                print(f"✓ 结构相似性：良好 (SSIM = {ssim:.4f})")
            else:
                print(f"⚠ 结构相似性：需改进 (SSIM = {ssim:.4f})")

    print("\n✓ 评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
