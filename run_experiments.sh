#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyclediff
cd /root/autodl-tmp/CycleDiff

echo "===== C-Gradient Experiment (c_grad=3.0) ====="
echo "Start: $(date)"
python train_uncond_ldm_cycle_swanlab.py --cfg ./configs/maps/translation_cgrad.yaml
echo "End: $(date)"

echo "===== Baseline Experiment (c_grad=0.0) ====="
echo "Start: $(date)"
python train_uncond_ldm_cycle_swanlab.py --cfg ./configs/maps/translation_baseline.yaml
echo "End: $(date)"

echo "===== All experiments completed ====="
