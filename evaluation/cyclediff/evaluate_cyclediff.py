#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CycleDiff 模型评估脚本
功能：
统一评估 CycleDiff 模型的「图像分量翻译」能力

评估流程：
1. 图像分量翻译质量：使用 reverse_q_sample_c_list_concat 从源域图像提取 C 列表，
   使用 Generator 逐时间步翻译 C，再使用目标域 LDM 从翻译后的 C 列表重建图像
2. 循环一致性：C_S → net_G_A → C_T → net_G_B → C_S'，评估循环重建质量
3. 恒等映射：评估 Generator 对自身域 C 的保持能力
4. C 分量逐时间步可视化：展示不同时间步 C_S 与翻译后 C_T 的对比
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description="Evaluate CycleDiff model - Image Component Translation")
    parser.add_argument("--cfg", type=str,
                        default="configs/maps/translation_C_disc_timestep_ode_2.yaml",
                        help="CycleDiff config file path")
    parser.add_argument("--ckpt", type=str,
                        default="results/maps/train_cycle_C_disc_G_blocks_12_maps_rsi2map/model-10.pt",
                        help="CycleDiff checkpoint path")
    parser.add_argument("--save_dir", type=str,
                        default="evaluation/cyclediff/res",
                        help="Results save directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--cal_metrics", action="store_true",
                        help="Calculate quantitative metrics")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--use_test_set", action="store_true",
                        help="Use test set instead of train set")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Use EMA weights")
    parser.add_argument("--direction", type=str, default="both",
                        choices=["A2B", "B2A", "both"],
                        help="Evaluation direction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    return args


def load_conf(config_file):
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf


def load_ema_state_dict(data, key):
    sd = data[key]
    new_sd = {}
    for k in sd.keys():
        if k.startswith("ema_model."):
            new_k = k[10:]
            new_sd[new_k] = sd[k]
    return new_sd


def load_cyclediff_model(cfg, ckpt_path, device, use_ema=True):
    model_cfg1 = cfg['model1']
    first_stage_cfg1 = model_cfg1['first_stage']
    first_stage_model1 = construct_class_by_name(**first_stage_cfg1)
    unet_cfg1 = model_cfg1['unet']
    unet1 = construct_class_by_name(**unet_cfg1)
    model_kwargs1 = {'model': unet1, 'auto_encoder': first_stage_model1, 'cfg': model_cfg1}
    model_kwargs1.update(model_cfg1)
    model1 = construct_class_by_name(**model_kwargs1)
    model_kwargs1.pop('model')
    model_kwargs1.pop('auto_encoder')

    model_cfg2 = cfg['model2']
    first_stage_cfg2 = model_cfg2['first_stage']
    first_stage_model2 = construct_class_by_name(**first_stage_cfg2)
    unet_cfg2 = model_cfg2['unet']
    unet2 = construct_class_by_name(**unet_cfg2)
    model_kwargs2 = {'model': unet2, 'auto_encoder': first_stage_model2, 'cfg': model_cfg2}
    model_kwargs2.update(model_cfg2)
    model2 = construct_class_by_name(**model_kwargs2)
    model_kwargs2.pop('model')
    model_kwargs2.pop('auto_encoder')

    net_G_A_cfg = cfg['net_G']
    net_G_A = construct_class_by_name(**net_G_A_cfg)
    net_G_B_cfg = cfg['net_G']
    net_G_B = construct_class_by_name(**net_G_B_cfg)

    if os.path.exists(ckpt_path):
        print(f"   从 {ckpt_path} 加载权重...")
        data = safe_torch_load(ckpt_path, map_location="cpu")
        if use_ema:
            print(f"   使用 EMA 权重")
            model1.load_state_dict(load_ema_state_dict(data, 'ema_d1'))
            model2.load_state_dict(load_ema_state_dict(data, 'ema_d2'))
            net_G_A.load_state_dict(load_ema_state_dict(data, 'ema_G_A'))
            net_G_B.load_state_dict(load_ema_state_dict(data, 'ema_G_B'))
        else:
            print(f"   使用原始权重")
            model1.load_state_dict(data['model1'])
            model2.load_state_dict(data['model2'])
            net_G_A.load_state_dict(data['net_G_A'])
            net_G_B.load_state_dict(data['net_G_B'])
        if 'model1' in data and 'scale_factor' in data['model1']:
            model1.scale_factor = data['model1']['scale_factor']
            model2.scale_factor = data['model2']['scale_factor']
        print(f"   scale_factor: model1={model1.scale_factor}, model2={model2.scale_factor}")
    else:
        print(f"   警告：权重文件不存在 {ckpt_path}，使用随机初始化")

    model1 = model1.to(device).eval()
    model2 = model2.to(device).eval()
    net_G_A = net_G_A.to(device).eval()
    net_G_B = net_G_B.to(device).eval()

    return model1, model2, net_G_A, net_G_B


def get_t_steps(model, device):
    step = 1. / model.sampling_timesteps
    rho = 1.
    step_indices = torch.arange(model.sampling_timesteps, dtype=torch.float32, device=device)
    t_steps = (model.sigma_max ** (1 / rho) + step_indices / (model.sampling_timesteps - 1) * (
            step - model.sigma_max ** (1 / rho))) ** rho
    t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))
    return t_steps


def translate_c_list(c_list, noise, net_G, t_steps, batch_size):
    target_input = []
    for i in range(len(c_list[:-1])):
        target_input.append(net_G(c_list[i], t_steps[i + 1].repeat((batch_size,))))
    target_input.append(net_G(c_list[-1], t_steps[-1].repeat((batch_size,))))
    target_input.append(noise)
    return target_input


def create_comparison_grid(images_list, labels, num_images=4):
    comparison = []
    for i in range(min(num_images, images_list[0].shape[0])):
        for imgs in images_list:
            img = imgs[i]
            if img.shape[-2:] != images_list[0][i].shape[-2:]:
                img = F.interpolate(img.unsqueeze(0), size=images_list[0][i].shape[-2:],
                                    mode='bilinear', align_corners=False).squeeze(0)
            comparison.append(img)
    comparison = torch.stack(comparison, dim=0)
    grid = tv.utils.make_grid(
        comparison,
        nrow=len(images_list),
        normalize=True,
        value_range=(0, 1),
        padding=2,
        pad_value=1.0
    )
    return grid


def normalize_c_to_display(c_tensor):
    c_min = c_tensor.flatten(1).min(dim=1, keepdim=True)[0][:, :, None, None]
    c_max = c_tensor.flatten(1).max(dim=1, keepdim=True)[0][:, :, None, None]
    c_norm = (c_tensor - c_min) / (c_max - c_min + 1e-8)
    return c_norm


def create_c_comparison_grid(c_list, num_images=4):
    comparison = []
    for i in range(min(num_images, c_list[0].shape[0])):
        for c_tensor in c_list:
            c_i = c_tensor[i:i + 1]
            c_norm = (c_i - c_i.min()) / (c_i.max() - c_i.min() + 1e-8)
            comparison.append(c_norm.squeeze(0))
    comparison = torch.stack(comparison, dim=0)
    grid = tv.utils.make_grid(
        comparison,
        nrow=len(c_list),
        normalize=False,
        padding=2,
        pad_value=1.0
    )
    return grid


def compute_c_diff_map(c_src, c_rec):
    diff = (c_src - c_rec).abs()
    diff = diff.mean(dim=1, keepdim=True)
    diff_min = diff.flatten(1).min(dim=1, keepdim=True)[0][:, :, None, None]
    diff_max = diff.flatten(1).max(dim=1, keepdim=True)[0][:, :, None, None]
    diff_norm = (diff - diff_min) / (diff_max - diff_min + 1e-8)
    diff_rgb = diff_norm.repeat(1, 3, 1, 1)
    return diff_rgb


def evaluate_c_translation(model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
                           save_dir, device, num_samples=50, direction="both"):
    model1.eval()
    model2.eval()
    net_G_A.eval()
    net_G_B.eval()

    results = {}

    if direction in ["A2B", "both"]:
        print("\n" + "=" * 60)
        print("评估 A→B 方向（RSI→Map）图像分量翻译质量")
        print("=" * 60)
        a2b_dir = os.path.join(save_dir, "A2B_translation")
        os.makedirs(os.path.join(a2b_dir, "source"), exist_ok=True)
        os.makedirs(os.path.join(a2b_dir, "translated"), exist_ok=True)
        os.makedirs(os.path.join(a2b_dir, "comparison"), exist_ok=True)

        t_steps1 = get_t_steps(model1, device)
        total_samples = 0
        comparison_grids = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_a):
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list, noise = model1.reverse_q_sample_c_list_concat(src_img)
                target_input = translate_c_list(c_list, noise, net_G_A, t_steps1, batch_size)
                pred_img = model2.sample_from_c_list(batch_size=batch_size, c_list=target_input)

                src_display = (src_img + 1.0) / 2.0

                for i in range(min(batch_size, num_samples - total_samples)):
                    tv.utils.save_image(src_display[i:i + 1],
                                        os.path.join(a2b_dir, "source", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(pred_img[i:i + 1],
                                        os.path.join(a2b_dir, "translated", f"sample_{total_samples + i:03d}.png"))

                n_compare = min(batch_size, 4)
                grid = create_comparison_grid(
                    [src_display[:n_compare], pred_img[:n_compare]],
                    ["Source (RSI)", "Translated (Map)"],
                    num_images=n_compare
                )
                comparison_grids.append(grid)

                total_samples += batch_size
                if batch_idx % 5 == 0:
                    print(f"  A→B 已处理 {min(total_samples, num_samples)}/{num_samples} 个样本")

        for idx, grid in enumerate(comparison_grids[:10]):
            tv.utils.save_image(grid, os.path.join(a2b_dir, "comparison", f"comparison_batch_{idx:03d}.png"))
        if comparison_grids:
            all_grids = torch.cat(comparison_grids[:5], dim=1)
            tv.utils.save_image(all_grids, os.path.join(a2b_dir, "comparison", "all_comparison.png"))

        print(f"✓ A→B 翻译结果已保存到：{a2b_dir}")
        results['A2B'] = {'source_path': os.path.join(a2b_dir, "source"),
                          'translated_path': os.path.join(a2b_dir, "translated"),
                          'num_samples': min(total_samples, num_samples)}

    if direction in ["B2A", "both"]:
        print("\n" + "=" * 60)
        print("评估 B→A 方向（Map→RSI）图像分量翻译质量")
        print("=" * 60)
        b2a_dir = os.path.join(save_dir, "B2A_translation")
        os.makedirs(os.path.join(b2a_dir, "source"), exist_ok=True)
        os.makedirs(os.path.join(b2a_dir, "translated"), exist_ok=True)
        os.makedirs(os.path.join(b2a_dir, "comparison"), exist_ok=True)

        t_steps2 = get_t_steps(model2, device)
        total_samples = 0
        comparison_grids = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_b):
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list, noise = model2.reverse_q_sample_c_list_concat(src_img)
                target_input = translate_c_list(c_list, noise, net_G_B, t_steps2, batch_size)
                pred_img = model1.sample_from_c_list(batch_size=batch_size, c_list=target_input)

                src_display = (src_img + 1.0) / 2.0

                for i in range(min(batch_size, num_samples - total_samples)):
                    tv.utils.save_image(src_display[i:i + 1],
                                        os.path.join(b2a_dir, "source", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(pred_img[i:i + 1],
                                        os.path.join(b2a_dir, "translated", f"sample_{total_samples + i:03d}.png"))

                n_compare = min(batch_size, 4)
                grid = create_comparison_grid(
                    [src_display[:n_compare], pred_img[:n_compare]],
                    ["Source (Map)", "Translated (RSI)"],
                    num_images=n_compare
                )
                comparison_grids.append(grid)

                total_samples += batch_size
                if batch_idx % 5 == 0:
                    print(f"  B→A 已处理 {min(total_samples, num_samples)}/{num_samples} 个样本")

        for idx, grid in enumerate(comparison_grids[:10]):
            tv.utils.save_image(grid, os.path.join(b2a_dir, "comparison", f"comparison_batch_{idx:03d}.png"))
        if comparison_grids:
            all_grids = torch.cat(comparison_grids[:5], dim=1)
            tv.utils.save_image(all_grids, os.path.join(b2a_dir, "comparison", "all_comparison.png"))

        print(f"✓ B→A 翻译结果已保存到：{b2a_dir}")
        results['B2A'] = {'source_path': os.path.join(b2a_dir, "source"),
                          'translated_path': os.path.join(b2a_dir, "translated"),
                          'num_samples': min(total_samples, num_samples)}

    return results


def evaluate_cycle_consistency(model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
                               save_dir, device, num_samples=50, direction="both"):
    model1.eval()
    model2.eval()
    net_G_A.eval()
    net_G_B.eval()

    results = {}

    if direction in ["A2B", "both"]:
        print("\n" + "=" * 60)
        print("评估 A→B→A 循环一致性")
        print("=" * 60)
        cycle_dir = os.path.join(save_dir, "cycle_ABA")
        os.makedirs(os.path.join(cycle_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(cycle_dir, "forward"), exist_ok=True)
        os.makedirs(os.path.join(cycle_dir, "reconstructed"), exist_ok=True)
        os.makedirs(os.path.join(cycle_dir, "comparison"), exist_ok=True)
        c_cycle_dir = os.path.join(cycle_dir, "c_space")
        os.makedirs(os.path.join(c_cycle_dir, "src_c"), exist_ok=True)
        os.makedirs(os.path.join(c_cycle_dir, "fwd_c"), exist_ok=True)
        os.makedirs(os.path.join(c_cycle_dir, "rec_c"), exist_ok=True)
        os.makedirs(os.path.join(c_cycle_dir, "diff_c"), exist_ok=True)
        os.makedirs(os.path.join(c_cycle_dir, "comparison"), exist_ok=True)

        t_steps1 = get_t_steps(model1, device)
        total_samples = 0
        comparison_grids = []
        c_comparison_grids = []
        cycle_l1_scores = []
        c_cycle_l1_scores = []
        c_cycle_ssim_scores = []
        c_cycle_psnr_scores = []
        c_cycle_lpips_scores = []
        cycle_ssim_scores = []
        cycle_psnr_scores = []
        cycle_lpips_scores = []
        from taming.modules.losses.lpips import LPIPS as LPIPSModel
        lpips_fn = LPIPSModel().eval().to(device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_a):
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list_src, noise_src = model1.reverse_q_sample_c_list_concat(src_img)
                target_input_fwd = translate_c_list(c_list_src, noise_src, net_G_A, t_steps1, batch_size)
                fwd_img = model2.sample_from_c_list(batch_size=batch_size, c_list=target_input_fwd)

                c_list_fwd, noise_fwd = model2.reverse_q_sample_c_list_concat(fwd_img)
                t_steps2 = get_t_steps(model2, device)
                target_input_bwd = translate_c_list(c_list_fwd, noise_fwd, net_G_B, t_steps2, batch_size)
                rec_img = model1.sample_from_c_list(batch_size=batch_size, c_list=target_input_bwd)

                src_display = (src_img + 1.0) / 2.0
                fwd_display = fwd_img
                rec_display = rec_img

                pixel_l1 = F.l1_loss(rec_display, src_display, reduction='none').mean(dim=[1, 2, 3])
                cycle_l1_scores.extend(pixel_l1.cpu().numpy().tolist())

                from util.mse_psnr_ssim_mssim import ssim as calc_ssim, psnr as calc_psnr
                ssim_val = calc_ssim(rec_display, src_display, data_range=1, size_average=False)
                cycle_ssim_scores.extend(ssim_val.cpu().numpy().tolist())
                psnr_val = calc_psnr(rec_display, src_display, data_range=1.0)
                cycle_psnr_scores.extend(psnr_val.cpu().numpy().tolist())
                lpips_val = lpips_fn(rec_display * 2.0 - 1.0, src_display * 2.0 - 1.0)
                cycle_lpips_scores.extend(lpips_val.flatten().cpu().numpy().tolist())

                src_latent = model1.scale_factor * model1.get_first_stage_encoding(
                    model1.first_stage_model.encode(src_img))
                rec_latent = model1.scale_factor * model1.get_first_stage_encoding(
                    model1.first_stage_model.encode(rec_img * 2.0 - 1.0))
                c_src = -1 * src_latent
                c_rec = -1 * rec_latent
                c_l1 = F.l1_loss(c_rec, c_src, reduction='none').mean(dim=[1, 2, 3])
                c_cycle_l1_scores.extend(c_l1.cpu().numpy().tolist())

                from util.mse_psnr_ssim_mssim import ssim as calc_ssim, psnr as calc_psnr
                c_ssim_val = calc_ssim(c_rec, c_src, data_range=float(c_src.max() - c_src.min()), size_average=False)
                c_cycle_ssim_scores.extend(c_ssim_val.cpu().numpy().tolist())
                c_psnr_val = calc_psnr(c_rec, c_src, data_range=float(c_src.max() - c_src.min()))
                c_cycle_psnr_scores.extend(c_psnr_val.cpu().numpy().tolist())

                with torch.no_grad():
                    z_src = -c_src / model1.scale_factor
                    z_rec = -c_rec / model1.scale_factor
                    decoded_src = model1.first_stage_model.decode(z_src.to(torch.float32))
                    decoded_rec = model1.first_stage_model.decode(z_rec.to(torch.float32))
                    decoded_src = (decoded_src + 1.0) * 0.5
                    decoded_rec = (decoded_rec + 1.0) * 0.5
                    decoded_src = torch.clamp(decoded_src, 0., 1.)
                    decoded_rec = torch.clamp(decoded_rec, 0., 1.)
                c_lpips_val = lpips_fn(decoded_rec * 2.0 - 1.0, decoded_src * 2.0 - 1.0)
                c_cycle_lpips_scores.extend(c_lpips_val.flatten().cpu().numpy().tolist())

                fwd_latent = model2.scale_factor * model2.get_first_stage_encoding(
                    model2.first_stage_model.encode(fwd_img * 2.0 - 1.0))
                c_fwd = -1 * fwd_latent

                c_src_display = normalize_c_to_display(c_src)
                c_fwd_display = normalize_c_to_display(c_fwd)
                c_rec_display = normalize_c_to_display(c_rec)
                c_diff_display = compute_c_diff_map(c_src, c_rec)

                for i in range(min(batch_size, num_samples - total_samples)):
                    tv.utils.save_image(src_display[i:i + 1],
                                        os.path.join(cycle_dir, "original", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(fwd_display[i:i + 1],
                                        os.path.join(cycle_dir, "forward", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(rec_display[i:i + 1],
                                        os.path.join(cycle_dir, "reconstructed", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(c_src_display[i:i + 1],
                                        os.path.join(c_cycle_dir, "src_c", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(c_fwd_display[i:i + 1],
                                        os.path.join(c_cycle_dir, "fwd_c", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(c_rec_display[i:i + 1],
                                        os.path.join(c_cycle_dir, "rec_c", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(c_diff_display[i:i + 1],
                                        os.path.join(c_cycle_dir, "diff_c", f"sample_{total_samples + i:03d}.png"))

                n_compare = min(batch_size, 4)
                grid = create_comparison_grid(
                    [src_display[:n_compare], fwd_display[:n_compare], rec_display[:n_compare]],
                    ["Original", "Forward (A→B)", "Reconstructed (A→B→A)"],
                    num_images=n_compare
                )
                comparison_grids.append(grid)

                c_grid = create_c_comparison_grid(
                    [c_src_display[:n_compare], c_fwd_display[:n_compare],
                     c_rec_display[:n_compare], c_diff_display[:n_compare]],
                    num_images=n_compare
                )
                c_comparison_grids.append(c_grid)

                total_samples += batch_size
                if batch_idx % 5 == 0:
                    print(f"  A→B→A 已处理 {min(total_samples, num_samples)}/{num_samples} 个样本")

        for idx, grid in enumerate(comparison_grids[:10]):
            tv.utils.save_image(grid, os.path.join(cycle_dir, "comparison", f"comparison_batch_{idx:03d}.png"))
        if comparison_grids:
            all_grids = torch.cat(comparison_grids[:5], dim=1)
            tv.utils.save_image(all_grids, os.path.join(cycle_dir, "comparison", "all_comparison.png"))

        for idx, grid in enumerate(c_comparison_grids[:10]):
            tv.utils.save_image(grid, os.path.join(c_cycle_dir, "comparison", f"c_comparison_batch_{idx:03d}.png"))
        if c_comparison_grids:
            all_c_grids = torch.cat(c_comparison_grids[:5], dim=1)
            tv.utils.save_image(all_c_grids, os.path.join(c_cycle_dir, "comparison", "all_c_comparison.png"))

        results['ABA'] = {
            'pixel_l1_mean': float(np.mean(cycle_l1_scores)),
            'pixel_l1_std': float(np.std(cycle_l1_scores)),
            'c_l1_mean': float(np.mean(c_cycle_l1_scores)),
            'c_l1_std': float(np.std(c_cycle_l1_scores)),
            'c_ssim_mean': float(np.mean(c_cycle_ssim_scores)),
            'c_ssim_std': float(np.std(c_cycle_ssim_scores)),
            'c_psnr_mean': float(np.mean(c_cycle_psnr_scores)),
            'c_psnr_std': float(np.std(c_cycle_psnr_scores)),
            'c_lpips_mean': float(np.mean(c_cycle_lpips_scores)),
            'c_lpips_std': float(np.std(c_cycle_lpips_scores)),
            'ssim_mean': float(np.mean(cycle_ssim_scores)),
            'ssim_std': float(np.std(cycle_ssim_scores)),
            'psnr_mean': float(np.mean(cycle_psnr_scores)),
            'psnr_std': float(np.std(cycle_psnr_scores)),
            'lpips_mean': float(np.mean(cycle_lpips_scores)),
            'lpips_std': float(np.std(cycle_lpips_scores)),
            'original_path': os.path.join(cycle_dir, "original"),
            'reconstructed_path': os.path.join(cycle_dir, "reconstructed"),
            'c_space_path': c_cycle_dir,
        }
        print(f"✓ A→B→A 循环一致性 - 像素L1: {results['ABA']['pixel_l1_mean']:.6f}, C空间L1: {results['ABA']['c_l1_mean']:.6f}")
        print(f"  像素域 SSIM: {results['ABA']['ssim_mean']:.4f}, PSNR: {results['ABA']['psnr_mean']:.2f} dB, LPIPS: {results['ABA']['lpips_mean']:.4f}")
        print(f"  C空间 SSIM: {results['ABA']['c_ssim_mean']:.4f}, PSNR: {results['ABA']['c_psnr_mean']:.2f} dB, LPIPS: {results['ABA']['c_lpips_mean']:.4f}")
        print(f"  C空间可视化已保存到：{c_cycle_dir}")

    if direction in ["B2A", "both"]:
        print("\n" + "=" * 60)
        print("评估 B→A→B 循环一致性")
        print("=" * 60)
        cycle_dir = os.path.join(save_dir, "cycle_BAB")
        os.makedirs(os.path.join(cycle_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(cycle_dir, "forward"), exist_ok=True)
        os.makedirs(os.path.join(cycle_dir, "reconstructed"), exist_ok=True)
        os.makedirs(os.path.join(cycle_dir, "comparison"), exist_ok=True)
        c_cycle_dir = os.path.join(cycle_dir, "c_space")
        os.makedirs(os.path.join(c_cycle_dir, "src_c"), exist_ok=True)
        os.makedirs(os.path.join(c_cycle_dir, "fwd_c"), exist_ok=True)
        os.makedirs(os.path.join(c_cycle_dir, "rec_c"), exist_ok=True)
        os.makedirs(os.path.join(c_cycle_dir, "diff_c"), exist_ok=True)
        os.makedirs(os.path.join(c_cycle_dir, "comparison"), exist_ok=True)

        t_steps2 = get_t_steps(model2, device)
        total_samples = 0
        comparison_grids = []
        c_comparison_grids = []
        cycle_l1_scores = []
        c_cycle_l1_scores = []
        c_cycle_ssim_scores = []
        c_cycle_psnr_scores = []
        c_cycle_lpips_scores = []
        cycle_ssim_scores = []
        cycle_psnr_scores = []
        cycle_lpips_scores = []
        from taming.modules.losses.lpips import LPIPS as LPIPSModel
        lpips_fn = LPIPSModel().eval().to(device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_b):
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list_src, noise_src = model2.reverse_q_sample_c_list_concat(src_img)
                target_input_fwd = translate_c_list(c_list_src, noise_src, net_G_B, t_steps2, batch_size)
                fwd_img = model1.sample_from_c_list(batch_size=batch_size, c_list=target_input_fwd)

                c_list_fwd, noise_fwd = model1.reverse_q_sample_c_list_concat(fwd_img)
                t_steps1 = get_t_steps(model1, device)
                target_input_bwd = translate_c_list(c_list_fwd, noise_fwd, net_G_A, t_steps1, batch_size)
                rec_img = model2.sample_from_c_list(batch_size=batch_size, c_list=target_input_bwd)

                src_display = (src_img + 1.0) / 2.0
                fwd_display = fwd_img
                rec_display = rec_img

                pixel_l1 = F.l1_loss(rec_display, src_display, reduction='none').mean(dim=[1, 2, 3])
                cycle_l1_scores.extend(pixel_l1.cpu().numpy().tolist())

                from util.mse_psnr_ssim_mssim import ssim as calc_ssim, psnr as calc_psnr
                ssim_val = calc_ssim(rec_display, src_display, data_range=1, size_average=False)
                cycle_ssim_scores.extend(ssim_val.cpu().numpy().tolist())
                psnr_val = calc_psnr(rec_display, src_display, data_range=1.0)
                cycle_psnr_scores.extend(psnr_val.cpu().numpy().tolist())
                lpips_val = lpips_fn(rec_display * 2.0 - 1.0, src_display * 2.0 - 1.0)
                cycle_lpips_scores.extend(lpips_val.flatten().cpu().numpy().tolist())

                src_latent = model2.scale_factor * model2.get_first_stage_encoding(
                    model2.first_stage_model.encode(src_img))
                rec_latent = model2.scale_factor * model2.get_first_stage_encoding(
                    model2.first_stage_model.encode(rec_img * 2.0 - 1.0))
                c_src = -1 * src_latent
                c_rec = -1 * rec_latent
                c_l1 = F.l1_loss(c_rec, c_src, reduction='none').mean(dim=[1, 2, 3])
                c_cycle_l1_scores.extend(c_l1.cpu().numpy().tolist())

                from util.mse_psnr_ssim_mssim import ssim as calc_ssim, psnr as calc_psnr
                c_ssim_val = calc_ssim(c_rec, c_src, data_range=float(c_src.max() - c_src.min()), size_average=False)
                c_cycle_ssim_scores.extend(c_ssim_val.cpu().numpy().tolist())
                c_psnr_val = calc_psnr(c_rec, c_src, data_range=float(c_src.max() - c_src.min()))
                c_cycle_psnr_scores.extend(c_psnr_val.cpu().numpy().tolist())

                with torch.no_grad():
                    z_src = -c_src / model2.scale_factor
                    z_rec = -c_rec / model2.scale_factor
                    decoded_src = model2.first_stage_model.decode(z_src.to(torch.float32))
                    decoded_rec = model2.first_stage_model.decode(z_rec.to(torch.float32))
                    decoded_src = (decoded_src + 1.0) * 0.5
                    decoded_rec = (decoded_rec + 1.0) * 0.5
                    decoded_src = torch.clamp(decoded_src, 0., 1.)
                    decoded_rec = torch.clamp(decoded_rec, 0., 1.)
                c_lpips_val = lpips_fn(decoded_rec * 2.0 - 1.0, decoded_src * 2.0 - 1.0)
                c_cycle_lpips_scores.extend(c_lpips_val.flatten().cpu().numpy().tolist())

                fwd_latent = model1.scale_factor * model1.get_first_stage_encoding(
                    model1.first_stage_model.encode(fwd_img * 2.0 - 1.0))
                c_fwd = -1 * fwd_latent

                c_src_display = normalize_c_to_display(c_src)
                c_fwd_display = normalize_c_to_display(c_fwd)
                c_rec_display = normalize_c_to_display(c_rec)
                c_diff_display = compute_c_diff_map(c_src, c_rec)

                for i in range(min(batch_size, num_samples - total_samples)):
                    tv.utils.save_image(src_display[i:i + 1],
                                        os.path.join(cycle_dir, "original", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(fwd_display[i:i + 1],
                                        os.path.join(cycle_dir, "forward", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(rec_display[i:i + 1],
                                        os.path.join(cycle_dir, "reconstructed", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(c_src_display[i:i + 1],
                                        os.path.join(c_cycle_dir, "src_c", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(c_fwd_display[i:i + 1],
                                        os.path.join(c_cycle_dir, "fwd_c", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(c_rec_display[i:i + 1],
                                        os.path.join(c_cycle_dir, "rec_c", f"sample_{total_samples + i:03d}.png"))
                    tv.utils.save_image(c_diff_display[i:i + 1],
                                        os.path.join(c_cycle_dir, "diff_c", f"sample_{total_samples + i:03d}.png"))

                n_compare = min(batch_size, 4)
                grid = create_comparison_grid(
                    [src_display[:n_compare], fwd_display[:n_compare], rec_display[:n_compare]],
                    ["Original", "Forward (B→A)", "Reconstructed (B→A→B)"],
                    num_images=n_compare
                )
                comparison_grids.append(grid)

                c_grid = create_c_comparison_grid(
                    [c_src_display[:n_compare], c_fwd_display[:n_compare],
                     c_rec_display[:n_compare], c_diff_display[:n_compare]],
                    num_images=n_compare
                )
                c_comparison_grids.append(c_grid)

                total_samples += batch_size
                if batch_idx % 5 == 0:
                    print(f"  B→A→B 已处理 {min(total_samples, num_samples)}/{num_samples} 个样本")

        for idx, grid in enumerate(comparison_grids[:10]):
            tv.utils.save_image(grid, os.path.join(cycle_dir, "comparison", f"comparison_batch_{idx:03d}.png"))
        if comparison_grids:
            all_grids = torch.cat(comparison_grids[:5], dim=1)
            tv.utils.save_image(all_grids, os.path.join(cycle_dir, "comparison", "all_comparison.png"))

        for idx, grid in enumerate(c_comparison_grids[:10]):
            tv.utils.save_image(grid, os.path.join(c_cycle_dir, "comparison", f"c_comparison_batch_{idx:03d}.png"))
        if c_comparison_grids:
            all_c_grids = torch.cat(c_comparison_grids[:5], dim=1)
            tv.utils.save_image(all_c_grids, os.path.join(c_cycle_dir, "comparison", "all_c_comparison.png"))

        results['BAB'] = {
            'pixel_l1_mean': float(np.mean(cycle_l1_scores)),
            'pixel_l1_std': float(np.std(cycle_l1_scores)),
            'c_l1_mean': float(np.mean(c_cycle_l1_scores)),
            'c_l1_std': float(np.std(c_cycle_l1_scores)),
            'c_ssim_mean': float(np.mean(c_cycle_ssim_scores)),
            'c_ssim_std': float(np.std(c_cycle_ssim_scores)),
            'c_psnr_mean': float(np.mean(c_cycle_psnr_scores)),
            'c_psnr_std': float(np.std(c_cycle_psnr_scores)),
            'c_lpips_mean': float(np.mean(c_cycle_lpips_scores)),
            'c_lpips_std': float(np.std(c_cycle_lpips_scores)),
            'ssim_mean': float(np.mean(cycle_ssim_scores)),
            'ssim_std': float(np.std(cycle_ssim_scores)),
            'psnr_mean': float(np.mean(cycle_psnr_scores)),
            'psnr_std': float(np.std(cycle_psnr_scores)),
            'lpips_mean': float(np.mean(cycle_lpips_scores)),
            'lpips_std': float(np.std(cycle_lpips_scores)),
            'original_path': os.path.join(cycle_dir, "original"),
            'reconstructed_path': os.path.join(cycle_dir, "reconstructed"),
            'c_space_path': c_cycle_dir,
        }
        print(f"✓ B→A→B 循环一致性 - 像素L1: {results['BAB']['pixel_l1_mean']:.6f}, C空间L1: {results['BAB']['c_l1_mean']:.6f}")
        print(f"  像素域 SSIM: {results['BAB']['ssim_mean']:.4f}, PSNR: {results['BAB']['psnr_mean']:.2f} dB, LPIPS: {results['BAB']['lpips_mean']:.4f}")
        print(f"  C空间 SSIM: {results['BAB']['c_ssim_mean']:.4f}, PSNR: {results['BAB']['c_psnr_mean']:.2f} dB, LPIPS: {results['BAB']['c_lpips_mean']:.4f}")
        print(f"  C空间可视化已保存到：{c_cycle_dir}")

    return results


def evaluate_identity(model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
                      save_dir, device, num_samples=50, direction="both"):
    model1.eval()
    model2.eval()
    net_G_A.eval()
    net_G_B.eval()

    results = {}

    if direction in ["A2B", "both"]:
        print("\n" + "=" * 60)
        print("评估恒等映射：net_G_B 应保持 A 域 C 不变")
        print("=" * 60)
        idt_dir = os.path.join(save_dir, "identity_A")
        os.makedirs(idt_dir, exist_ok=True)

        t_steps1 = get_t_steps(model1, device)
        total_samples = 0
        idt_l1_scores = []
        idt_cos_scores = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_a):
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list, noise = model1.reverse_q_sample_c_list_concat(src_img)

                idt_c_list = []
                for i in range(len(c_list[:-1])):
                    idt_c = net_G_B(c_list[i], t_steps1[i + 1].repeat((batch_size,)))
                    idt_c_list.append(idt_c)
                idt_c_last = net_G_B(c_list[-1], t_steps1[-1].repeat((batch_size,)))
                idt_c_list.append(idt_c_last)

                for i in range(len(c_list)):
                    l1 = F.l1_loss(idt_c_list[i], c_list[i], reduction='none').mean(dim=[1, 2, 3])
                    idt_l1_scores.extend(l1.cpu().numpy().tolist())
                    cos_sim = F.cosine_similarity(idt_c_list[i].flatten(1), c_list[i].flatten(1), dim=1)
                    idt_cos_scores.extend(cos_sim.cpu().numpy().tolist())

                target_input = idt_c_list + [noise]
                idt_img = model1.sample_from_c_list(batch_size=batch_size, c_list=target_input)

                src_display = (src_img + 1.0) / 2.0
                n_compare = min(batch_size, 4)
                grid = create_comparison_grid(
                    [src_display[:n_compare], idt_img[:n_compare]],
                    ["Original (A)", "Identity (A→B→A)"],
                    num_images=n_compare
                )
                tv.utils.save_image(grid, os.path.join(idt_dir, f"identity_batch_{batch_idx:03d}.png"))

                total_samples += batch_size
                if batch_idx % 5 == 0:
                    print(f"  A域恒等映射 已处理 {min(total_samples, num_samples)}/{num_samples} 个样本")

        results['identity_A'] = {
            'c_l1_mean': float(np.mean(idt_l1_scores)),
            'c_l1_std': float(np.std(idt_l1_scores)),
            'c_cos_mean': float(np.mean(idt_cos_scores)),
            'c_cos_std': float(np.std(idt_cos_scores)),
        }
        print(f"✓ A域恒等映射 - C空间L1: {results['identity_A']['c_l1_mean']:.6f}, "
              f"余弦相似度: {results['identity_A']['c_cos_mean']:.4f}")

    if direction in ["B2A", "both"]:
        print("\n" + "=" * 60)
        print("评估恒等映射：net_G_A 应保持 B 域 C 不变")
        print("=" * 60)
        idt_dir = os.path.join(save_dir, "identity_B")
        os.makedirs(idt_dir, exist_ok=True)

        t_steps2 = get_t_steps(model2, device)
        total_samples = 0
        idt_l1_scores = []
        idt_cos_scores = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_b):
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list, noise = model2.reverse_q_sample_c_list_concat(src_img)

                idt_c_list = []
                for i in range(len(c_list[:-1])):
                    idt_c = net_G_A(c_list[i], t_steps2[i + 1].repeat((batch_size,)))
                    idt_c_list.append(idt_c)
                idt_c_last = net_G_A(c_list[-1], t_steps2[-1].repeat((batch_size,)))
                idt_c_list.append(idt_c_last)

                for i in range(len(c_list)):
                    l1 = F.l1_loss(idt_c_list[i], c_list[i], reduction='none').mean(dim=[1, 2, 3])
                    idt_l1_scores.extend(l1.cpu().numpy().tolist())
                    cos_sim = F.cosine_similarity(idt_c_list[i].flatten(1), c_list[i].flatten(1), dim=1)
                    idt_cos_scores.extend(cos_sim.cpu().numpy().tolist())

                target_input = idt_c_list + [noise]
                idt_img = model2.sample_from_c_list(batch_size=batch_size, c_list=target_input)

                src_display = (src_img + 1.0) / 2.0
                n_compare = min(batch_size, 4)
                grid = create_comparison_grid(
                    [src_display[:n_compare], idt_img[:n_compare]],
                    ["Original (B)", "Identity (B→A→B)"],
                    num_images=n_compare
                )
                tv.utils.save_image(grid, os.path.join(idt_dir, f"identity_batch_{batch_idx:03d}.png"))

                total_samples += batch_size
                if batch_idx % 5 == 0:
                    print(f"  B域恒等映射 已处理 {min(total_samples, num_samples)}/{num_samples} 个样本")

        results['identity_B'] = {
            'c_l1_mean': float(np.mean(idt_l1_scores)),
            'c_l1_std': float(np.std(idt_l1_scores)),
            'c_cos_mean': float(np.mean(idt_cos_scores)),
            'c_cos_std': float(np.std(idt_cos_scores)),
        }
        print(f"✓ B域恒等映射 - C空间L1: {results['identity_B']['c_l1_mean']:.6f}, "
              f"余弦相似度: {results['identity_B']['c_cos_mean']:.4f}")

    return results


def visualize_c_components(model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
                           save_dir, device, num_samples=4, direction="both"):
    model1.eval()
    model2.eval()
    net_G_A.eval()
    net_G_B.eval()

    if direction in ["A2B", "both"]:
        print("\n" + "=" * 60)
        print("可视化 A→B 方向 C 分量逐时间步翻译")
        print("=" * 60)
        vis_dir = os.path.join(save_dir, "A2B_c_visualization")
        os.makedirs(vis_dir, exist_ok=True)

        t_steps1 = get_t_steps(model1, device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_a):
                if batch_idx >= 1:
                    break
                src_img = batch['image'].to(device)
                batch_size = min(src_img.shape[0], num_samples)

                c_list, noise = model1.reverse_q_sample_c_list_concat(src_img)

                translated_c_list = []
                for i in range(len(c_list[:-1])):
                    translated_c = net_G_A(c_list[i], t_steps1[i + 1].repeat((batch_size,)))
                    translated_c_list.append(translated_c)
                translated_c_last = net_G_A(c_list[-1], t_steps1[-1].repeat((batch_size,)))
                translated_c_list.append(translated_c_last)

                n_timesteps = len(c_list)
                n_show = min(n_timesteps, 10)
                step_interval = max(1, n_timesteps // n_show)

                for sample_idx in range(batch_size):
                    rows = []
                    for t_idx in range(0, n_timesteps, step_interval):
                        c_s = c_list[t_idx][sample_idx:sample_idx + 1]
                        c_t = translated_c_list[t_idx][sample_idx:sample_idx + 1]

                        c_s_norm = (c_s - c_s.min()) / (c_s.max() - c_s.min() + 1e-8)
                        c_t_norm = (c_t - c_t.min()) / (c_t.max() - c_t.min() + 1e-8)

                        rows.append(c_s_norm)
                        rows.append(c_t_norm)

                    if rows:
                        grid = tv.utils.make_grid(torch.cat(rows, dim=0), nrow=2, padding=2, pad_value=1.0)
                        tv.utils.save_image(grid, os.path.join(vis_dir, f"sample_{sample_idx:03d}_c_components.png"))

                print(f"  ✓ C分量可视化已保存到：{vis_dir}")

    if direction in ["B2A", "both"]:
        print("\n" + "=" * 60)
        print("可视化 B→A 方向 C 分量逐时间步翻译")
        print("=" * 60)
        vis_dir = os.path.join(save_dir, "B2A_c_visualization")
        os.makedirs(vis_dir, exist_ok=True)

        t_steps2 = get_t_steps(model2, device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_b):
                if batch_idx >= 1:
                    break
                src_img = batch['image'].to(device)
                batch_size = min(src_img.shape[0], num_samples)

                c_list, noise = model2.reverse_q_sample_c_list_concat(src_img)

                translated_c_list = []
                for i in range(len(c_list[:-1])):
                    translated_c = net_G_B(c_list[i], t_steps2[i + 1].repeat((batch_size,)))
                    translated_c_list.append(translated_c)
                translated_c_last = net_G_B(c_list[-1], t_steps2[-1].repeat((batch_size,)))
                translated_c_list.append(translated_c_last)

                n_timesteps = len(c_list)
                n_show = min(n_timesteps, 10)
                step_interval = max(1, n_timesteps // n_show)

                for sample_idx in range(batch_size):
                    rows = []
                    for t_idx in range(0, n_timesteps, step_interval):
                        c_s = c_list[t_idx][sample_idx:sample_idx + 1]
                        c_t = translated_c_list[t_idx][sample_idx:sample_idx + 1]

                        c_s_norm = (c_s - c_s.min()) / (c_s.max() - c_s.min() + 1e-8)
                        c_t_norm = (c_t - c_t.min()) / (c_t.max() - c_t.min() + 1e-8)

                        rows.append(c_s_norm)
                        rows.append(c_t_norm)

                    if rows:
                        grid = tv.utils.make_grid(torch.cat(rows, dim=0), nrow=2, padding=2, pad_value=1.0)
                        tv.utils.save_image(grid, os.path.join(vis_dir, f"sample_{sample_idx:03d}_c_components.png"))

                print(f"  ✓ C分量可视化已保存到：{vis_dir}")


def calculate_c_space_metrics(model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
                              device, num_samples=50, direction="both"):
    model1.eval()
    model2.eval()
    net_G_A.eval()
    net_G_B.eval()

    results = {}

    if direction in ["A2B", "both"]:
        print("\n正在计算 A→B 方向 C 空间翻译指标...")
        t_steps1 = get_t_steps(model1, device)
        total_samples = 0
        c_l1_scores = []
        c_l2_scores = []
        c_cos_scores = []

        with torch.no_grad():
            for batch in dataloader_a:
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list, noise = model1.reverse_q_sample_c_list_concat(src_img)

                translated_c_list = []
                for i in range(len(c_list[:-1])):
                    translated_c = net_G_A(c_list[i], t_steps1[i + 1].repeat((batch_size,)))
                    translated_c_list.append(translated_c)
                translated_c_last = net_G_A(c_list[-1], t_steps1[-1].repeat((batch_size,)))
                translated_c_list.append(translated_c_last)

                src_latent = model1.scale_factor * model1.get_first_stage_encoding(
                    model1.first_stage_model.encode(src_img))
                c_src = -1 * src_latent

                target_input = translated_c_list + [noise]
                pred_img = model2.sample_from_c_list(batch_size=batch_size, c_list=target_input)
                pred_latent = model2.scale_factor * model2.get_first_stage_encoding(
                    model2.first_stage_model.encode(pred_img * 2.0 - 1.0))
                c_trg = -1 * pred_latent

                c_src_flat = c_src.flatten(1)
                c_trg_flat = c_trg.flatten(1)

                l1 = F.l1_loss(c_trg, c_src, reduction='none').mean(dim=[1, 2, 3])
                l2 = F.mse_loss(c_trg, c_src, reduction='none').mean(dim=[1, 2, 3]).sqrt()
                cos_sim = F.cosine_similarity(c_trg_flat, c_src_flat, dim=1)

                c_l1_scores.extend(l1.cpu().numpy().tolist())
                c_l2_scores.extend(l2.cpu().numpy().tolist())
                c_cos_scores.extend(cos_sim.cpu().numpy().tolist())

                total_samples += batch_size

        results['A2B_c_metrics'] = {
            'c_l1_mean': float(np.mean(c_l1_scores)),
            'c_l1_std': float(np.std(c_l1_scores)),
            'c_l2_mean': float(np.mean(c_l2_scores)),
            'c_l2_std': float(np.std(c_l2_scores)),
            'c_cos_mean': float(np.mean(c_cos_scores)),
            'c_cos_std': float(np.std(c_cos_scores)),
        }
        print(f"  A→B C空间指标 - L1: {results['A2B_c_metrics']['c_l1_mean']:.6f}, "
              f"L2: {results['A2B_c_metrics']['c_l2_mean']:.6f}, "
              f"余弦相似度: {results['A2B_c_metrics']['c_cos_mean']:.4f}")

    if direction in ["B2A", "both"]:
        print("\n正在计算 B→A 方向 C 空间翻译指标...")
        t_steps2 = get_t_steps(model2, device)
        total_samples = 0
        c_l1_scores = []
        c_l2_scores = []
        c_cos_scores = []

        with torch.no_grad():
            for batch in dataloader_b:
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list, noise = model2.reverse_q_sample_c_list_concat(src_img)

                translated_c_list = []
                for i in range(len(c_list[:-1])):
                    translated_c = net_G_B(c_list[i], t_steps2[i + 1].repeat((batch_size,)))
                    translated_c_list.append(translated_c)
                translated_c_last = net_G_B(c_list[-1], t_steps2[-1].repeat((batch_size,)))
                translated_c_list.append(translated_c_last)

                src_latent = model2.scale_factor * model2.get_first_stage_encoding(
                    model2.first_stage_model.encode(src_img))
                c_src = -1 * src_latent

                target_input = translated_c_list + [noise]
                pred_img = model1.sample_from_c_list(batch_size=batch_size, c_list=target_input)
                pred_latent = model1.scale_factor * model1.get_first_stage_encoding(
                    model1.first_stage_model.encode(pred_img * 2.0 - 1.0))
                c_trg = -1 * pred_latent

                c_src_flat = c_src.flatten(1)
                c_trg_flat = c_trg.flatten(1)

                l1 = F.l1_loss(c_trg, c_src, reduction='none').mean(dim=[1, 2, 3])
                l2 = F.mse_loss(c_trg, c_src, reduction='none').mean(dim=[1, 2, 3]).sqrt()
                cos_sim = F.cosine_similarity(c_trg_flat, c_src_flat, dim=1)

                c_l1_scores.extend(l1.cpu().numpy().tolist())
                c_l2_scores.extend(l2.cpu().numpy().tolist())
                c_cos_scores.extend(cos_sim.cpu().numpy().tolist())

                total_samples += batch_size

        results['B2A_c_metrics'] = {
            'c_l1_mean': float(np.mean(c_l1_scores)),
            'c_l1_std': float(np.std(c_l1_scores)),
            'c_l2_mean': float(np.mean(c_l2_scores)),
            'c_l2_std': float(np.std(c_l2_scores)),
            'c_cos_mean': float(np.mean(c_cos_scores)),
            'c_cos_std': float(np.std(c_cos_scores)),
        }
        print(f"  B→A C空间指标 - L1: {results['B2A_c_metrics']['c_l1_mean']:.6f}, "
              f"L2: {results['B2A_c_metrics']['c_l2_mean']:.6f}, "
              f"余弦相似度: {results['B2A_c_metrics']['c_cos_mean']:.4f}")

    return results


def calculate_translation_metrics(save_dir, direction="both"):
    metrics = {}

    if direction in ["A2B", "both"]:
        a2b_dir = os.path.join(save_dir, "A2B_translation")
        source_path = os.path.join(a2b_dir, "source")
        translated_path = os.path.join(a2b_dir, "translated")
        if os.path.exists(source_path) and os.path.exists(translated_path):
            print("\n正在计算 A→B 翻译重建指标...")
            mse = calculate_mse(translated_path, source_path)
            psnr = calculate_psnr(translated_path, source_path)
            ssim = calculate_ssim(translated_path, source_path)
            msssim = calculate_msssim(translated_path, source_path)
            metrics['A2B_reconstruction'] = {
                'mse': mse, 'psnr': psnr, 'ssim': ssim, 'ms_ssim': msssim
            }
            print(f"  MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, MS-SSIM: {msssim:.4f}")

    if direction in ["B2A", "both"]:
        b2a_dir = os.path.join(save_dir, "B2A_translation")
        source_path = os.path.join(b2a_dir, "source")
        translated_path = os.path.join(b2a_dir, "translated")
        if os.path.exists(source_path) and os.path.exists(translated_path):
            print("\n正在计算 B→A 翻译重建指标...")
            mse = calculate_mse(translated_path, source_path)
            psnr = calculate_psnr(translated_path, source_path)
            ssim = calculate_ssim(translated_path, source_path)
            msssim = calculate_msssim(translated_path, source_path)
            metrics['B2A_reconstruction'] = {
                'mse': mse, 'psnr': psnr, 'ssim': ssim, 'ms_ssim': msssim
            }
            print(f"  MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, MS-SSIM: {msssim:.4f}")

    return metrics


def calculate_lpips_metrics(model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
                            device, num_samples=50, direction="both"):
    from taming.modules.losses.lpips import LPIPS

    lpips_model = LPIPS().eval().to(device)
    results = {}

    if direction in ["A2B", "both"]:
        print("\n正在计算 A→B 方向 LPIPS...")
        t_steps1 = get_t_steps(model1, device)
        lpips_scores = []
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader_a:
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list, noise = model1.reverse_q_sample_c_list_concat(src_img)
                target_input = translate_c_list(c_list, noise, net_G_A, t_steps1, batch_size)
                pred_img = model2.sample_from_c_list(batch_size=batch_size, c_list=target_input)

                pred_norm = pred_img * 2.0 - 1.0
                lpips_score = lpips_model(src_img, pred_norm)
                lpips_scores.append(lpips_score.mean().item())

                total_samples += batch_size

        results['A2B_lpips'] = {
            'mean': float(np.mean(lpips_scores)),
            'std': float(np.std(lpips_scores))
        }
        print(f"  A→B LPIPS: {results['A2B_lpips']['mean']:.6f} ± {results['A2B_lpips']['std']:.6f}")

    if direction in ["B2A", "both"]:
        print("\n正在计算 B→A 方向 LPIPS...")
        t_steps2 = get_t_steps(model2, device)
        lpips_scores = []
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader_b:
                if total_samples >= num_samples:
                    break
                src_img = batch['image'].to(device)
                batch_size = src_img.shape[0]

                c_list, noise = model2.reverse_q_sample_c_list_concat(src_img)
                target_input = translate_c_list(c_list, noise, net_G_B, t_steps2, batch_size)
                pred_img = model1.sample_from_c_list(batch_size=batch_size, c_list=target_input)

                pred_norm = pred_img * 2.0 - 1.0
                lpips_score = lpips_model(src_img, pred_norm)
                lpips_scores.append(lpips_score.mean().item())

                total_samples += batch_size

        results['B2A_lpips'] = {
            'mean': float(np.mean(lpips_scores)),
            'std': float(np.std(lpips_scores))
        }
        print(f"  B→A LPIPS: {results['B2A_lpips']['mean']:.6f} ± {results['B2A_lpips']['std']:.6f}")

    return results


def save_metrics(metrics, save_dir):
    metrics_path = os.path.join(save_dir, "evaluation_metrics.txt")

    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("CycleDiff 模型评估报告 - 图像分量翻译能力\n")
        f.write("=" * 60 + "\n\n")

        f.write("定性评估:\n")
        for key in ['A2B', 'B2A']:
            trans_key = f'{key}_translation'
            if trans_key in metrics.get('translation', {}):
                info = metrics['translation'][trans_key]
                f.write(f"  {key} 翻译:\n")
                f.write(f"    源域图像：{info.get('source_path', 'N/A')}\n")
                f.write(f"    翻译图像：{info.get('translated_path', 'N/A')}\n")
        for key in ['ABA', 'BAB']:
            cycle_key = f'cycle_{key}'
            if cycle_key in metrics.get('cycle', {}):
                info = metrics['cycle'][cycle_key]
                f.write(f"  循环一致性 {key}:\n")
                f.write(f"    原始图像：{info.get('original_path', 'N/A')}\n")
                f.write(f"    重建图像：{info.get('reconstructed_path', 'N/A')}\n")
        f.write("\n")

        f.write("定量评估:\n")
        for category, cat_metrics in metrics.items():
            if category in ['translation']:
                continue
            if isinstance(cat_metrics, dict):
                for key, value in cat_metrics.items():
                    if isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for k, v in value.items():
                            if isinstance(v, float):
                                f.write(f"    {k}: {v:.6f}\n")
                            else:
                                f.write(f"    {k}: {v}\n")
                    elif isinstance(value, float):
                        f.write(f"  {key}: {value:.6f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("评估完成!\n")

    print(f"✓ 评估报告已保存到：{metrics_path}")


def main(args):
    print("=" * 60)
    print("CycleDiff 模型评估 - 图像分量翻译能力")
    print("=" * 60)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(f"\n1. 加载配置文件：{args.cfg}")
    cfg = load_conf(args.cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 优先从配置文件的 sampler.ckpt_path 读取模型路径
    ckpt_path = cfg.get('sampler', {}).get('ckpt_path', args.ckpt)
    if ckpt_path and ckpt_path != args.ckpt:
        print(f"   从配置文件 sampler.ckpt_path 读取模型路径: {ckpt_path}")
    else:
        ckpt_path = args.ckpt
        print(f"   使用命令行参数指定的模型路径: {ckpt_path}")

    print(f"\n2. 加载 CycleDiff 模型：{ckpt_path}")
    model1, model2, net_G_A, net_G_B = load_cyclediff_model(cfg, ckpt_path, device, args.use_ema)

    print(f"\n3. 加载数据集")
    data_test_cfg_a = cfg.get('data_test', {})
    data_test_cfg_b = cfg.get('data_test2', {})

    if args.use_test_set:
        if data_test_cfg_a:
            data_test_cfg_a['split'] = 'test'
        if data_test_cfg_b:
            data_test_cfg_b['split'] = 'test'
        print(f"   使用测试集进行评估")
    else:
        data_cfg = cfg.get('data', {})
        data_cfg_a = dict(data_cfg)
        data_cfg_a['split'] = 'train'
        data_cfg_a['image_size'] = [256, 256]
        data_cfg_a.pop('batch_size', None)
        data_cfg_a['class_name'] = 'ddm.data.Single_dataset'
        data_cfg_a['datafolder_name'] = data_cfg.get('source_folder_name', 'class_RSI')
        data_test_cfg_a = data_cfg_a

        data_cfg_b = dict(data_cfg)
        data_cfg_b['split'] = 'train'
        data_cfg_b['image_size'] = [256, 256]
        data_cfg_b.pop('batch_size', None)
        data_cfg_b['class_name'] = 'ddm.data.Single_dataset'
        data_cfg_b['datafolder_name'] = data_cfg.get('target_folder_name', 'class_Map')
        data_test_cfg_b = data_cfg_b
        print(f"   使用训练集进行评估")

    dataset_a = construct_class_by_name(**data_test_cfg_a)
    dataset_b = construct_class_by_name(**data_test_cfg_b)
    dataloader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    dataloader_b = DataLoader(dataset_b, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f"   A域数据集大小：{len(dataset_a)}")
    print(f"   B域数据集大小：{len(dataset_b)}")

    print(f"\n4. 创建结果保存目录：{args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    all_metrics = {}

    print("\n" + "=" * 60)
    print("评估1：图像分量翻译质量")
    print("=" * 60)
    translation_results = evaluate_c_translation(
        model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
        args.save_dir, device, args.num_samples, args.direction
    )
    all_metrics['translation'] = translation_results

    print("\n" + "=" * 60)
    print("评估2：循环一致性")
    print("=" * 60)
    cycle_results = evaluate_cycle_consistency(
        model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
        args.save_dir, device, args.num_samples, args.direction
    )
    all_metrics['cycle'] = cycle_results

    print("\n" + "=" * 60)
    print("评估3：恒等映射")
    print("=" * 60)
    identity_results = evaluate_identity(
        model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
        args.save_dir, device, args.num_samples, args.direction
    )
    all_metrics['identity'] = identity_results

    print("\n" + "=" * 60)
    print("评估4：C 分量逐时间步可视化")
    print("=" * 60)
    visualize_c_components(
        model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
        args.save_dir, device, num_samples=4, direction=args.direction
    )

    if args.cal_metrics:
        print("\n" + "=" * 60)
        print("定量评估")
        print("=" * 60)

        recon_metrics = calculate_translation_metrics(args.save_dir, args.direction)
        all_metrics['reconstruction'] = recon_metrics

        c_metrics = calculate_c_space_metrics(
            model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
            device, args.num_samples, args.direction
        )
        all_metrics['c_space'] = c_metrics

        try:
            lpips_results = calculate_lpips_metrics(
                model1, model2, net_G_A, net_G_B, dataloader_a, dataloader_b,
                device, args.num_samples, args.direction
            )
            all_metrics['lpips'] = lpips_results
        except Exception as e:
            print(f"  LPIPS 计算失败：{e}")

    save_metrics(all_metrics, args.save_dir)

    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)

    if 'cycle' in all_metrics:
        for key in ['ABA', 'BAB']:
            if key in all_metrics['cycle']:
                cycle_info = all_metrics['cycle'][key]
                pixel_l1 = cycle_info.get('pixel_l1_mean', float('inf'))
                c_l1 = cycle_info.get('c_l1_mean', float('inf'))
                ssim_val = cycle_info.get('ssim_mean', 0)
                psnr_val = cycle_info.get('psnr_mean', 0)
                lpips_val = cycle_info.get('lpips_mean', 1)
                c_ssim_val = cycle_info.get('c_ssim_mean', 0)
                c_psnr_val = cycle_info.get('c_psnr_mean', 0)
                c_lpips_val = cycle_info.get('c_lpips_mean', 1)

                if pixel_l1 < 0.05:
                    l1_grade = "优秀"
                elif pixel_l1 < 0.10:
                    l1_grade = "良好"
                elif pixel_l1 < 0.15:
                    l1_grade = "一般"
                else:
                    l1_grade = "需改进"

                if ssim_val > 0.90:
                    ssim_grade = "优秀"
                elif ssim_val > 0.75:
                    ssim_grade = "良好"
                elif ssim_val > 0.60:
                    ssim_grade = "一般"
                else:
                    ssim_grade = "需改进"

                if psnr_val > 30:
                    psnr_grade = "优秀"
                elif psnr_val > 25:
                    psnr_grade = "良好"
                elif psnr_val > 20:
                    psnr_grade = "一般"
                else:
                    psnr_grade = "需改进"

                if lpips_val < 0.10:
                    lpips_grade = "优秀"
                elif lpips_val < 0.20:
                    lpips_grade = "良好"
                elif lpips_val < 0.35:
                    lpips_grade = "一般"
                else:
                    lpips_grade = "需改进"

                if c_l1 < 0.05:
                    c_l1_grade = "优秀"
                elif c_l1 < 0.10:
                    c_l1_grade = "良好"
                elif c_l1 < 0.20:
                    c_l1_grade = "一般"
                else:
                    c_l1_grade = "需改进"

                if c_ssim_val > 0.90:
                    c_ssim_grade = "优秀"
                elif c_ssim_val > 0.75:
                    c_ssim_grade = "良好"
                elif c_ssim_val > 0.60:
                    c_ssim_grade = "一般"
                else:
                    c_ssim_grade = "需改进"

                if c_psnr_val > 30:
                    c_psnr_grade = "优秀"
                elif c_psnr_val > 25:
                    c_psnr_grade = "良好"
                elif c_psnr_val > 20:
                    c_psnr_grade = "一般"
                else:
                    c_psnr_grade = "需改进"

                if c_lpips_val < 0.10:
                    c_lpips_grade = "优秀"
                elif c_lpips_val < 0.20:
                    c_lpips_grade = "良好"
                elif c_lpips_val < 0.35:
                    c_lpips_grade = "一般"
                else:
                    c_lpips_grade = "需改进"

                grades = {"优秀": 4, "良好": 3, "一般": 2, "需改进": 1}
                pixel_grade_num = min(
                    grades[l1_grade], grades[ssim_grade],
                    grades[psnr_grade], grades[lpips_grade]
                )
                pixel_grade = {v: k for k, v in grades.items()}[pixel_grade_num]
                c_grade_num = min(
                    grades[c_l1_grade], grades[c_ssim_grade],
                    grades[c_psnr_grade], grades[c_lpips_grade]
                )
                c_grade = {v: k for k, v in grades.items()}[c_grade_num]
                overall_grade_num = min(pixel_grade_num, c_grade_num)
                overall_grade = {v: k for k, v in grades.items()}[overall_grade_num]

                print(f"循环一致性 {key} 综合评级：{overall_grade}")
                print(f"  ── 像素域 ──")
                print(f"  L1:   {pixel_l1:.6f} [{l1_grade}]")
                print(f"  SSIM: {ssim_val:.4f} [{ssim_grade}]")
                print(f"  PSNR: {psnr_val:.2f} dB [{psnr_grade}]")
                print(f"  LPIPS:{lpips_val:.4f} [{lpips_grade}]")
                print(f"  ── C空间 ──")
                print(f"  L1:   {c_l1:.6f} [{c_l1_grade}]")
                print(f"  SSIM: {c_ssim_val:.4f} [{c_ssim_grade}]")
                print(f"  PSNR: {c_psnr_val:.2f} dB [{c_psnr_grade}]")
                print(f"  LPIPS:{c_lpips_val:.4f} [{c_lpips_grade}]")

    if 'identity' in all_metrics:
        for key in ['identity_A', 'identity_B']:
            if key in all_metrics['identity']:
                idt_info = all_metrics['identity'][key]
                cos_sim = idt_info.get('c_cos_mean', 0)
                if cos_sim > 0.95:
                    print(f"✓ 恒等映射 {key}：优秀 (余弦相似度 = {cos_sim:.4f})")
                elif cos_sim > 0.90:
                    print(f"✓ 恒等映射 {key}：良好 (余弦相似度 = {cos_sim:.4f})")
                else:
                    print(f"⚠ 恒等映射 {key}：需改进 (余弦相似度 = {cos_sim:.4f})")

    if 'reconstruction' in all_metrics:
        for key in ['A2B_reconstruction', 'B2A_reconstruction']:
            if key in all_metrics['reconstruction']:
                rec = all_metrics['reconstruction'][key]
                if 'psnr' in rec:
                    psnr = rec['psnr']
                    if psnr > 30:
                        print(f"✓ 翻译重建 {key}：优秀 (PSNR > 30 dB)")
                    elif psnr > 25:
                        print(f"✓ 翻译重建 {key}：良好 (PSNR = {psnr:.2f} dB)")
                    else:
                        print(f"⚠ 翻译重建 {key}：需改进 (PSNR = {psnr:.2f} dB)")

    print("\n✓ 评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
