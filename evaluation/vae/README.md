# VAE 模型评估工具

本目录包含用于评估 VAE（变分自编码器）模型质量的脚本。

## 📁 目录结构

```
evaluation/vae/
├── evaluate_vae.py      # 完整评估脚本（定性 + 定量）
├── quick_eval.py        # 快速评估脚本（仅定性）
├── run_eval.sh          # 交互式运行脚本
└── README.md            # 本文档
```

## 🚀 使用方法

### 方法 1：完整评估（推荐）

```bash
cd /root/autodl-tmp/CycleDiff

python evaluation/vae/evaluate_vae.py \
    --cfg configs/rsi2map/map_ae_kl_256x256_d4.yaml \
    --ckpt results/map_ae_kl_256x256_d4/model-10.pt \
    --save_dir evaluation/vae/res/map \
    --batch_size 16 \
    --num_samples 50 \
    --cal_metrics \
    --use_test_set
```

```Shell
python evaluation/vae/evaluate_vae.py \
    --cfg configs/rsi2map/rsi_ae_kl_256x256_d4.yaml \
    --ckpt results/rsi_ae_kl_256x256_d4/model-6.pt \
    --save_dir evaluation/vae/res/rsi \
    --batch_size 4 \
    --num_samples 50 \
    --cal_metrics \
    --use_test_set
```

**输出**：

- `evaluation/vae/res/original/` - 原始图像
- `evaluation/vae/res/reconstructed/` - 重建图像
- `evaluation/vae/res/comparison/` - 对比图像
- `evaluation/vae/res/evaluation_metrics.txt` - 评估指标

### 方法 2：快速评估

```bash
python evaluation/vae/quick_eval.py \
    --ckpt results/map_ae_kl_256x256_d4/model-10.pt \
    --save_dir evaluation/vae/res/quick_eval \
    --num_samples 20
```

### 方法 3：交互式运行

```bash
bash evaluation/vae/run_eval.sh
```

脚本会列出所有可用的模型文件，让你选择要评估的 checkpoint。

## 📊 评估指标

### 定性评估

- **重建图像对比**：直观查看原始图像 vs 重建图像

### 定量评估

| 指标          | 含义         | 优秀    | 良好        | 需改进   |
| ----------- | ---------- | ----- | --------- | ----- |
| **MSE**     | 均方误差       | <0.01 | 0.01-0.05 | >0.05 |
| **PSNR**    | 峰值信噪比 (dB) | >30   | 25-30     | <25   |
| **SSIM**    | 结构相似性      | >0.95 | 0.90-0.95 | <0.90 |
| **MS-SSIM** | 多尺度 SSIM   | >0.98 | 0.95-0.98 | <0.95 |
| **LPIPS**   | 感知相似度      | <0.05 | 0.05-0.15 | >0.15 |
| **KL 散度**   | 潜在空间分布     | <0.01 | 0.01-0.1  | >0.1  |

## 🔧 参数说明

### evaluate\_vae.py

| 参数               | 默认值                                           | 说明               |
| ---------------- | --------------------------------------------- | ---------------- |
| `--cfg`          | configs/rsi2map/map\_ae\_kl\_256x256\_d4.yaml | 配置文件路径           |
| `--ckpt`         | results/map\_ae\_kl\_256x256\_d4/model-10.pt  | 模型 checkpoint 路径 |
| `--save_dir`     | evaluation/vae/res                            | 结果保存目录           |
| `--batch_size`   | 16                                            | 批大小              |
| `--num_samples`  | 50                                            | 评估样本数量           |
| `--cal_metrics`  | False                                         | 是否计算定量指标         |
| `--use_test_set` | False                                         | 是否使用测试集          |

### quick\_eval.py

| 参数              | 默认值                                          | 说明               |
| --------------- | -------------------------------------------- | ---------------- |
| `--ckpt`        | results/map\_ae\_kl\_256x256\_d4/model-10.pt | 模型 checkpoint 路径 |
| `--save_dir`    | evaluation/vae/res/quick\_eval               | 结果保存目录           |
| `--batch_size`  | 8                                            | 批大小              |
| `--num_samples` | 20                                           | 评估样本数量           |

## 📝 示例输出

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
    mean: 0.008234
    std: 0.002156
  mse: 0.023456
  psnr: 28.45
  ssim: 0.9234
  ms_ssim: 0.9567
  lpips:
    mean: 0.087654
    std: 0.023456

==================================================
评估完成!
```

### 评估总结示例

```
评估总结
==================================================
✓ VAE 质量：良好 (PSNR = 28.45 dB)
✓ 结构相似性：良好 (SSIM = 0.9234)
✓ 潜在空间分布：优秀 (KL < 0.01)

✓ 评估完成！
==================================================
```

## 🎯 评估不同 checkpoint

```bash
# 评估 model-1.pt
python evaluation/vae/evaluate_vae.py --ckpt results/map_ae_kl_256x256_d4/model-1.pt

# 评估 model-5.pt
python evaluation/vae/evaluate_vae.py --ckpt results/map_ae_kl_256x256_d4/model-5.pt

# 评估 model-10.pt（最终模型）
python evaluation/vae/evaluate_vae.py --ckpt results/map_ae_kl_256x256_d4/model-10.pt
```

## 💡 使用技巧

1. **快速预览**：先用 `quick_eval.py` 快速查看重建质量
2. **完整评估**：确认模型后，用 `evaluate_vae.py` 进行完整评估
3. **对比分析**：评估多个 checkpoint，选择最佳模型
4. **测试集评估**：使用 `--use_test_set` 在测试集上评估泛化能力

## ⚠️ 注意事项

1. 确保在 CycleDiff 项目的 conda 环境中运行
2. 评估前确认模型文件存在
3. 定量评估需要较长时间，建议先用少量样本快速评估
4. 使用 `--use_test_set` 前确保配置文件中有 test 数据

## 📧 问题反馈

如有问题，请检查：

- 模型路径是否正确
- 数据集路径是否在配置文件中正确设置
- GPU 显存是否足够（可减小 `--batch_size`）

