#!/bin/bash
# VAE 模型评估运行脚本

# 设置路径
CKPT_DIR="/root/autodl-tmp/CycleDiff/results/map_ae_kl_256x256_d4"
SAVE_DIR="/root/autodl-tmp/CycleDiff/evaluation/vae/res"
CONFIG="/root/autodl-tmp/CycleDiff/configs/rsi2map/map_ae_kl_256x256_d4.yaml"

# 选择 checkpoint
echo "可用的模型文件:"
ls -1 $CKPT_DIR/model-*.pt | sort -V

echo ""
read -p "请输入要评估的模型文件编号 (默认: 10): " MODEL_NUM
MODEL_NUM=${MODEL_NUM:-10}

CKPT="$CKPT_DIR/model-${MODEL_NUM}.pt"

echo ""
echo "======================================"
echo "VAE 模型评估"
echo "======================================"
echo "配置文件：$CONFIG"
echo "模型文件：$CKPT"
echo "保存目录：$SAVE_DIR"
echo "======================================"
echo ""

# 运行评估
cd /root/autodl-tmp/CycleDiff
python evaluation/vae/evaluate_vae.py \
    --cfg $CONFIG \
    --ckpt $CKPT \
    --save_dir $SAVE_DIR \
    --batch_size 16 \
    --num_samples 50 \
    --cal_metrics \
    --use_test_set

echo ""
echo "评估结果已保存到：$SAVE_DIR"
echo "查看对比图像：$SAVE_DIR/comparison/all_comparison.png"
echo "查看评估指标：$SAVE_DIR/evaluation_metrics.txt"
