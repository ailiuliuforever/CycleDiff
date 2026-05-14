#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LDM 模型评估脚本
功能：
统一评估 LDM 模型的「图像分量提取」能力和「基于提取的图像分量重建RGB图像」的能力

评估流程：
1. 图像分量提取：使用 reverse_q_sample_c_list_concat 从图像提取 C 列表和最终噪声状态 x_t
2. 图像重建：使用模型内置的 sample_from_c_list 方法，从提取的 C 列表和 x_t 重建 RGB 图像
   （参考 translation_uncond_ldm_cycle.py 中将翻译后的图像分量生成 RGB 图像的方法）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import yaml
import argparse
import torchvision as tv
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name, safe_torch_load
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
    # Use 100 sampling steps (matching paper) for better reconstruction
    model.sampling_timesteps = 100
    return model


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
    print("  采样流程：提取C列表和x_t → 使用模型内置 sample_from_c_list 从C列表重建RGB图像")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if total_samples >= num_samples:
                break

            img = batch['image'].to(device)
            batch_size = img.shape[0]

            c_list, x_t = model.reverse_q_sample_c_list_concat(img)

            x_rec = model.sample_from_c_list(batch_size=batch_size, c_list=list(c_list) + [x_t])

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


def visualize_c_components(model, dataloader, save_dir, device, num_samples=4):
    """可视化 LDM 模型在不同时间步提取的图像分量（C列表）"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    model.eval()
    vis_dir = os.path.join(save_dir, "c_visualization")
    os.makedirs(vis_dir, exist_ok=True)

    print("正在可视化 C 分量逐时间步...")
    print("  采样流程：提取C列表 → 对每个时间步的C分量做min-max归一化 → 拼接可视化")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:
                break

            img = batch['image'].to(device)
            batch_size = min(img.shape[0], num_samples)

            c_list, x_t = model.reverse_q_sample_c_list_concat(img)

            n_timesteps = len(c_list)
            n_show = min(n_timesteps, 10)
            step_interval = max(1, n_timesteps // n_show)

            for sample_idx in range(batch_size):
                rows = []
                t_indices = []
                for t_idx in range(0, n_timesteps, step_interval):
                    c = c_list[t_idx][sample_idx:sample_idx + 1]
                    c_norm = (c - c.min()) / (c.max() - c.min() + 1e-8)
                    rows.append(c_norm)
                    t_indices.append(t_idx)

                if rows:
                    # 使用 make_grid 生成纵向排列的图像 (nrow=1 表示每行1个，即纵向排列)
                    grid = tv.utils.make_grid(
                        torch.cat(rows, dim=0),
                        nrow=1,
                        padding=2,
                        pad_value=1.0
                    )

                    # 转换为 PIL Image
                    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(grid_np)

                    n_subplots = len(rows)
                    img_width = pil_img.width
                    img_height = pil_img.height

                    # make_grid(nrow=1) 的排列方式：
                    # 图像在垂直方向堆叠，每个子图之间有 padding
                    # 子图实际高度 = (img_height - padding * (n_subplots + 1)) / n_subplots
                    padding = 2
                    subplot_height = (img_height - padding * (n_subplots + 1)) // n_subplots
                    subplot_width = img_width - padding * 2

                    # 创建带标注的新图像
                    label_height = 20
                    new_img_height = img_height + label_height * n_subplots
                    new_img = Image.new('RGB', (img_width, new_img_height), (255, 255, 255))

                    draw = ImageDraw.Draw(new_img)

                    # 尝试加载字体
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                    except:
                        font = ImageFont.load_default()

                    # 逐个处理每个子图：先画标注，再贴图像
                    for i, t_idx in enumerate(t_indices):
                        # 计算当前子图的位置
                        label_y_start = i * (subplot_height + padding * 2 + label_height)
                        img_y_start = label_y_start + label_height

                        # 在原始 grid 中裁剪出当前子图
                        # make_grid 的排列：从上到下依次是 rows[0], rows[1], ...
                        src_y_start = padding + i * (subplot_height + padding)
                        src_y_end = src_y_start + subplot_height
                        subplot = pil_img.crop((padding, src_y_start, padding + subplot_width, src_y_end))

                        # 在新图像中粘贴标注和子图
                        draw.text((padding + 5, label_y_start + 2), f"t_idx: {t_idx}", fill=(0, 0, 0), font=font)
                        new_img.paste(subplot, (padding, img_y_start))

                    new_img.save(os.path.join(vis_dir, f"sample_{sample_idx:03d}_c_components.png"))

            print(f"  ✓ C分量可视化已保存到：{vis_dir}")
            print(f"    共 {n_timesteps} 个时间步，展示了 {len(rows)} 个时间步的分量")


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

            c_list, x_t = model.reverse_q_sample_c_list_concat(img)
            x_rec = model.sample_from_c_list(batch_size=img.shape[0], c_list=list(c_list) + [x_t])

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
    # Override image_size to 256 to match model's expected latent resolution (64x64 at 4x downsample)
    data_cfg['image_size'] = [256, 256]

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

    print("\n" + "=" * 60)
    print("评估：C 分量逐时间步可视化")
    print("=" * 60)
    visualize_c_components(model, dataloader, args.save_dir, device, num_samples=4)

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
