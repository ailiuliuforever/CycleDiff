#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LDM 模型快速评估脚本
评估从图像分量（C列表）重建原始图像的能力（仅定性评估）
保存双对比图：原始 vs 重建

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
import torchvision as tv
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name, safe_torch_load, unnormalize_to_zero_to_one
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Quick LDM evaluation - Reconstruction from C list")
    parser.add_argument("--cfg", type=str,
                        default="/root/autodl-tmp/CycleDiff/configs/maps/map_ddm_const4_ldm_unet6_114_ode_2.yaml",
                        help="LDM config file path")
    parser.add_argument("--ckpt", type=str,
                        default="/root/autodl-tmp/CycleDiff/results/maps/ddm_const_uncond_unet_ldm_map/model-10.pt",
                        help="LDM checkpoint path")
    parser.add_argument("--save_dir", type=str,
                        default="/root/autodl-tmp/CycleDiff/evaluation/ldm/res/quick_eval",
                        help="Results save directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    return args


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


def main(args):
    print("=" * 60)
    print("LDM 快速评估 - 从图像分量重建原始图像")
    print("=" * 60)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(f"\n加载配置：{args.cfg}")
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_cfg = cfg['model']
    data_cfg = cfg['data']
    data_cfg['split'] = 'train'

    first_stage_cfg = model_cfg['first_stage']
    first_stage_model = construct_class_by_name(**first_stage_cfg)

    unet_cfg = model_cfg['unet']
    unet = construct_class_by_name(**unet_cfg)

    model_kwargs = {'model': unet, 'auto_encoder': first_stage_model, 'cfg': model_cfg}
    model_kwargs.update(model_cfg)
    model = construct_class_by_name(**model_kwargs)
    model_kwargs.pop('model')
    model_kwargs.pop('auto_encoder')

    print(f"加载模型：{args.ckpt}")
    if os.path.exists(args.ckpt):
        data = safe_torch_load(args.ckpt, map_location="cpu")
        if 'model' in data:
            model.load_state_dict(data['model'])
            if 'scale_factor' in data['model']:
                model.scale_factor = data['model']['scale_factor']
        else:
            model.load_state_dict(data)
        print(f"✓ 成功加载权重, scale_factor: {model.scale_factor}")
    else:
        print(f"✗ 权重文件不存在：{args.ckpt}")
        return

    model = model.cuda()
    model.eval()

    dataset = construct_class_by_name(**data_cfg)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"保存目录：{args.save_dir}")

    print(f"\n评估 {args.num_samples} 个样本...")
    print("采样流程：从噪声开始 → UNet预测noise → 用预提取C替换UNet预测的C → 逐步去噪")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * args.batch_size >= args.num_samples:
                break

            img = batch['image'].cuda()

            c_list, _ = model.reverse_q_sample_c_list_concat(img)

            # 从 C 列表重建
            x_reconstructed = sample_from_c_list_correct(model, img.shape[0], c_list, img.device)

            img_display = (img + 1.0) / 2.0

            for i in range(img.shape[0]):
                comparison = tv.utils.make_grid(
                    [img_display[i], x_reconstructed[i]],
                    nrow=2,
                    normalize=False,
                    padding=5,
                    pad_value=1.0
                )
                tv.utils.save_image(
                    comparison,
                    os.path.join(args.save_dir, f"sample_{batch_idx * args.batch_size + i:03d}.png")
                )

            if batch_idx % 3 == 0:
                print(f"  已处理 {batch_idx + 1} 个 batch")

    print(f"\n✓ 评估完成！")
    print(f"查看结果：{args.save_dir}")
    print(f"每张图格式：[原始] [重建]")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
