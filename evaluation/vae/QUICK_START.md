# VAE 模型评估 - 快速开始指南

## ✅ 已创建的文件

```
evaluation/vae/
├── evaluate_vae.py      ✓ 完整评估脚本
├── quick_eval.py        ✓ 快速评估脚本  
├── run_eval.sh          ✓ 交互式运行脚本
└── README.md            ✓ 详细文档
```

## 🎯 快速使用（3 个命令）

### 1️⃣ 快速查看重建质量（推荐先试这个）

```bash
cd /root/autodl-tmp/CycleDiff
python evaluation/vae/quick_eval.py
```

**输出位置**：`evaluation/vae/res/quick_eval/`

---

### 2️⃣ 完整评估（包含所有指标）

```bash
cd /root/autodl-tmp/CycleDiff
python evaluation/vae/evaluate_vae.py \
    --ckpt results/map_ae_kl_256x256_d4/model-10.pt \
    --cal_metrics
```

**输出位置**：`evaluation/vae/res/`

**评估指标**：
- ✅ MSE（均方误差）
- ✅ PSNR（峰值信噪比）
- ✅ SSIM（结构相似性）
- ✅ MS-SSIM（多尺度结构相似性）
- ✅ LPIPS（感知相似度）
- ✅ KL 散度（潜在空间质量）

---

### 3️⃣ 交互式选择模型

```bash
cd /root/autodl-tmp/CycleDiff
bash evaluation/vae/run_eval.sh
```

脚本会自动列出所有可用的模型文件让你选择。

---

## 📊 可用的模型文件

根据 `/root/autodl-tmp/CycleDiff/results/map_ae_kl_256x256_d4/` 目录：

```
model-1.pt    ← 训练 5000 步
model-2.pt    ← 训练 10000 步
model-3.pt    ← 训练 15000 步
model-4.pt    ← 训练 20000 步
model-5.pt    ← 训练 25000 步
model-6.pt    ← 训练 30000 步
model-7.pt    ← 训练 35000 步
model-8.pt    ← 训练 40000 步
model-9.pt    ← 训练 45000 步
model-10.pt   ← 训练 50000 步（最终模型）✓ 推荐评估
```

---

## 📁 评估结果结构

运行评估后，结果将保存在以下位置：

```
evaluation/vae/res/
├── original/              # 原始图像
│   ├── sample_000.png
│   ├── sample_001.png
│   └── ...
├── reconstructed/         # VAE 重建图像
│   ├── sample_000.png
│   ├── sample_001.png
│   └── ...
├── comparison/            # 对比图（原始 + 重建并排）
│   ├── comparison_batch_000.png
│   ├── comparison_batch_001.png
│   ├── all_comparison.png  ← 综合对比图（最直观）
│   └── ...
├── quick_eval/            # 快速评估结果
│   └── sample_*.png
└── evaluation_metrics.txt  # 评估指标报告
```

---

## 🔍 评估指标解读

### 优秀 VAE 的标准

| 指标 | 优秀标准 | 含义 |
|------|----------|------|
| **PSNR** | >30 dB | 重建质量非常高 |
| **SSIM** | >0.95 | 结构保持非常好 |
| **MS-SSIM** | >0.98 | 多尺度结构相似 |
| **LPIPS** | <0.05 | 感知差异很小 |
| **KL 散度** | <0.01 | 潜在空间接近标准正态分布 |

### 评估报告示例

```
VAE 模型评估报告
==================================================

定性评估:
  - 原始图像：evaluation/vae/res/original/
  - 重建图像：evaluation/vae/res/reconstructed/
  - 对比图像：evaluation/vae/res/comparison/

定量评估:
  kl_divergence:
    mean: 0.008234  ✓ 优秀
  mse: 0.023456     ✓ 良好
  psnr: 28.45 dB    ✓ 良好
  ssim: 0.9234      ✓ 良好
  ms_ssim: 0.9567   ✓ 良好
  lpips: 0.087654   ✓ 良好

==================================================
评估总结
==================================================
✓ VAE 质量：良好 (PSNR = 28.45 dB)
✓ 结构相似性：良好 (SSIM = 0.9234)
✓ 潜在空间分布：优秀 (KL < 0.01)
```

---

## 💡 使用建议

1. **先用 quick_eval.py 快速预览**
   - 只需几秒钟
   - 直观查看重建效果
   - 决定是否需要深入评估

2. **再用 evaluate_vae.py 完整评估**
   - 计算所有定量指标
   - 生成详细报告
   - 用于论文/报告

3. **对比不同 checkpoint**
   ```bash
   # 评估早期模型
   python evaluate_vae.py --ckpt results/map_ae_kl_256x256_d4/model-5.pt
   
   # 评估最终模型
   python evaluate_vae.py --ckpt results/map_ae_kl_256x256_d4/model-10.pt
   ```

4. **在测试集上评估**
   ```bash
   python evaluate_vae.py --use_test_set
   ```

---

## ⚡ 一键评估所有模型

```bash
cd /root/autodl-tmp/CycleDiff

for i in {1..10}; do
    echo "评估 model-$i.pt..."
    python evaluation/vae/evaluate_vae.py \
        --ckpt results/map_ae_kl_256x256_d4/model-$i.pt \
        --save_dir evaluation/vae/res/model_$i \
        --num_samples 20 \
        --cal_metrics
done
```

---

## 🐛 常见问题

### Q: 显存不足怎么办？
A: 减小 `--batch_size` 参数：
```bash
python evaluate_vae.py --batch_size 4
```

### Q: 评估太慢怎么办？
A: 减少样本数量：
```bash
python evaluate_vae.py --num_samples 10
```

### Q: 如何只查看重建图像，不计算指标？
A: 去掉 `--cal_metrics` 参数：
```bash
python evaluate_vae.py  # 不进行定量评估
```

### Q: 如何评估 RSI 域的 VAE？
A: 修改配置文件路径：
```bash
python evaluate_vae.py \
    --cfg configs/rsi2map/rsi_ae_kl_256x256_d4.yaml \
    --ckpt results/rsi_ae_kl_256x256_d4/model-10.pt
```

---

## 📞 需要帮助？

查看详细文档：`cat evaluation/vae/README.md`

---

**Leader，评估脚本已经全部创建完成！** 🎉

你现在可以：
1. 运行 `python evaluation/vae/quick_eval.py` 快速查看 model-10.pt 的重建效果
2. 运行 `python evaluation/vae/evaluate_vae.py --cal_metrics` 获取完整评估报告
3. 查看 `evaluation/vae/README.md` 了解详细用法

所有评估结果将保存在 `/root/autodl-tmp/CycleDiff/evaluation/vae/res/` 目录下。
