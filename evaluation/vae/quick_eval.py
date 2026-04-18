#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE 模型快速评估脚本
快速查看模型的重建质量（仅定性评估）
"""

import torch
import yaml
import argparse
import os
import torchvision as tv
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Quick VAE evaluation")
    parser.add_argument("--ckpt", type=str, 
                        default="/root/autodl-tmp/CycleDiff/results/map_ae_kl_256x256_d4/model-10.pt",
                        help="VAE checkpoint path")
    parser.add_argument("--save_dir", type=str, 
                        default="/root/autodl-tmp/CycleDiff/evaluation/vae/res/quick_eval",
                        help="Results save directory")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples")
    args = parser.parse_args()
    return args


def main(args):
    print("=" * 50)
    print("VAE 快速评估")
    print("=" * 50)
    
    # 配置
    config_path = "/root/autodl-tmp/CycleDiff/configs/rsi2map/map_ae_kl_256x256_d4.yaml"
    
    print(f"\n加载配置：{config_path}")
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    
    # 加载模型
    print(f"加载模型：{args.ckpt}")
    model = construct_class_by_name(**model_cfg)
    
    if os.path.exists(args.ckpt):
        model.init_from_ckpt(args.ckpt)
        print(f"✓ 成功加载权重")
    else:
        print(f"✗ 权重文件不存在：{args.ckpt}")
        return
    
    model = model.cuda()
    model.eval()
    
    # 加载数据
    data_cfg['split'] = 'train'
    dataset = construct_class_by_name(**data_cfg)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # 保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"保存目录：{args.save_dir}")
    
    # 评估
    print(f"\n评估 {args.num_samples} 个样本...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * args.batch_size >= args.num_samples:
                break
            
            img = batch['image'].cuda()
            
            # 重建
            xrec, _ = model(img)
            
            # 保存对比图
            for i in range(img.shape[0]):
                comparison = tv.utils.make_grid(
                    [img[i], xrec[i]], 
                    nrow=2, 
                    normalize=True, 
                    value_range=(-1, 1),
                    padding=5,
                    pad_value=1.0
                )
                tv.utils.save_image(
                    comparison, 
                    os.path.join(args.save_dir, f"sample_{batch_idx * args.batch_size + i:03d}.png")
                )
            
            if batch_idx % 5 == 0:
                print(f"  已处理 {batch_idx + 1} 个 batch")
    
    print(f"\n✓ 评估完成！")
    print(f"查看结果：{args.save_dir}")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
