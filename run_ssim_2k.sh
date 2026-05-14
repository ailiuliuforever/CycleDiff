#!/bin/bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate cyclediff

echo "============================================"
echo "  CycleDiff cgrad 2000-step training (SSIM)"
echo "  Started at: $(date)"
echo "============================================"

echo ""
echo "Training with C-space multi-scale SSIM loss..."
python train_uncond_ldm_cycle_swanlab.py --cfg configs/maps/translation_cgrad.yaml

echo ""
echo "Training done. Running A2B C-component visualization..."
python evaluation/cyclediff/quick_a2b_c_vis.py

echo ""
echo "============================================"
echo "  Done at: $(date)"
echo "============================================"
