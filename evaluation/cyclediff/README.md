# CycleDiff 评估工具使用说明

## 概述

`evaluate_cyclediff.py` 是 CycleDiff 模型的评估工具，专门用于评估模型的\*\*「图像分量（C）翻译能力」\*\*。

在 CycleDiff 中，图像分量 C 是扩散过程中的确定性漂移项（`C = -x_0`），Generator 的核心作用是将源域的 C\_S 映射为目标域的 C\_T。本评估工具从多个维度量化这一核心能力。

## 功能特点

### 两种评估模式

#### 配对评估模式（`--paired`）

适用于**有配对数据**的场景（如测试集）。翻译结果将与目标域 GT 对比，能真实反映翻译质量。

- 数据要求：源域和目标域图像按文件名一一配对
- 对比对象：翻译结果 vs **目标域 GT**
- 适用指标：MSE/PSNR/SSIM/MS-SSIM、LPIPS、C 空间 L1/L2/余弦相似度
- 适用场景：最终测试报告、模型性能对比

#### 非配对评估模式（默认）

适用于**无配对数据**的场景（如训练集）。翻译结果与源域图像对比，主要用于验证循环一致性和身份映射。

- 数据要求：仅需源域图像
- 对比对象：翻译结果 vs **源域图像**
- 适用指标：循环一致性、恒等映射
- 适用场景：训练过程监控、调试、循环一致性验证

> **注意**：翻译质量指标（MSE/PSNR/SSIM/LPIPS）在非配对模式下与源域对比，对于图像翻译任务无实际意义，因为翻译结果本应不同于源域。这些指标仅在配对模式下能真实反映翻译质量。

### 四大评估模块

1. **C 翻译质量评估** (`evaluate_c_translation`)
   - 从源域图像提取 C 列表（`reverse_q_sample_c_list_concat`）
   - 使用 Generator 逐时间步翻译 C（`C_T = net_G(C_S, t)`）
   - 从翻译后的 C 列表在目标域采样重建图像
   - 支持 A→B（RSI→Map）和 B→A（Map→RSI）双向评估
2. **循环一致性评估** (`evaluate_cycle_consistency`)
   - 评估 C 空间的循环一致性：C\_S → C\_T → C\_S'
   - 计算像素级循环重建质量
   - 计算 C 空间的 L1 距离
3. **恒等映射评估** (`evaluate_identity`)
   - 评估 Generator 对自身域 C 的保持能力
   - 理想情况下：`net_G_B(C_S, t) ≈ C_S`
   - 计算 C 空间的 L1 距离和余弦相似度
4. **C 分量逐时间步可视化** (`visualize_c_components`)
   - 可视化不同时间步的 C\_S 和翻译后的 C\_T
   - 展示 Generator 如何在不同噪声水平下翻译 C 分量

### 定量指标

| 指标类型      | 具体指标                  | 含义            |
| --------- | --------------------- | ------------- |
| **像素级重建** | MSE、PSNR、SSIM、MS-SSIM | 翻译图像与源域图像的相似性 |
| **感知质量**  | LPIPS                 | 感知距离（越低越好）    |
| **C 空间**  | L1 距离、L2 距离、余弦相似度     | C 分量翻译的保真度    |
| **循环一致性** | 像素 L1、C 空间 L1         | 循环重建的误差       |

## 安装要求

确保在 `cyclediff` conda 环境中运行：

```bash
conda activate cyclediff
cd /root/autodl-tmp/CycleDiff
```

## 使用方法

### 基本用法

```bash
# 使用测试集进行评估（仅定性评估）
python evaluation/cyclediff/evaluate_cyclediff.py --use_test_set

# 完整评估（包含定量指标）
python evaluation/cyclediff/evaluate_cyclediff.py --use_test_set --cal_metrics
```

### 命令行参数详解

```
optional arguments:
  -h, --help            显示帮助信息并退出
  --cfg CFG             CycleDiff 配置文件路径
                        默认：configs/maps/translation_C_disc_timestep_ode_2.yaml
  --ckpt CKPT           CycleDiff checkpoint 路径
                        默认：results/maps/train_cycle_C_disc_G_blocks_12_maps_rsi2map/model-10.pt
  --save_dir SAVE_DIR   结果保存目录
                        默认：evaluation/cyclediff/res
  --batch_size BATCH_SIZE
                        评估时的批大小
                        默认：4
  --cal_metrics         是否计算定量指标（MSE/PSNR/SSIM/LPIPS 等）
                        默认：不计算
  --num_samples NUM_SAMPLES
                        评估的样本数量
                        默认：50
  --use_test_set        使用测试集而非训练集
                        默认：使用训练集
  --use_ema             使用 EMA 权重
                        默认：False
  --direction {A2B,B2A,both}
                        评估方向：
                        - A2B: 仅评估 A→B（RSI→Map）
                        - B2A: 仅评估 B→A（Map→RSI）
                        - both: 双向评估
                        默认：both
  --paired              使用配对数据进行评估
                        配对模式下，翻译结果将与目标域 GT 对比
                        非配对模式下，翻译结果将与源域图像对比
                        默认：False（非配对模式）
  --seed SEED           随机种子
                        默认：42
```

### 使用示例

#### 1. 快速定性评估（A→B 方向）

```bash
python evaluation/cyclediff/evaluate_cyclediff.py \
  --direction A2B \
  --num_samples 10 \
  --use_test_set
```

#### 2. 完整定量评估（双向，配对模式）

**推荐**：使用配对数据进行最终评估，翻译质量指标与目标域 GT 对比。

```bash
python evaluation/cyclediff/evaluate_cyclediff.py \
  --direction both \
  --num_samples 50 \
  --batch_size 4 \
  --cal_metrics \
  --use_test_set \
  --paired
```

#### 3. 完整定量评估（双向，非配对模式）

用于验证循环一致性和身份映射（翻译质量指标与源域对比，仅供参考）。

```bash
python evaluation/cyclediff/evaluate_cyclediff.py \
  --direction both \
  --num_samples 50 \
  --batch_size 4 \
  --cal_metrics \
  --use_test_set
```

#### 3. 评估特定 checkpoint

```bash
python evaluation/cyclediff/evaluate_cyclediff.py \
  --ckpt results/maps/train_cycle_C_disc_G_blocks_12_maps_rsi2map/model-8.pt \
  --save_dir evaluation/cyclediff/res_model8 \
  --cal_metrics \
  --use_test_set
```

#### 4. 仅评估循环一致性

```bash
python evaluation/cyclediff/evaluate_cyclediff.py \
  --direction A2B \
  --num_samples 20 \
  --use_test_set
# 循环一致性结果会在终端输出，并保存在 cycle_ABA/ 目录
```

## 输出结果说明

### 目录结构

运行评估后，`save_dir` 目录下会生成以下文件：

```
evaluation/cyclediff/res/
├── A2B_translation/           # A→B 翻译结果
│   ├── source/                # 源域图像（RSI）
│   ├── translated/            # 翻译后图像（Map）
│   ├── target_gt/             # 目标域 GT（Map，仅配对模式）
│   └── comparison/            # 对比图（源域 vs 翻译 vs 目标域GT，配对模式）
├── B2A_translation/           # B→A 翻译结果
│   ├── source/                # 源域图像（Map）
│   ├── translated/            # 翻译后图像（RSI）
│   ├── target_gt/             # 目标域 GT（RSI，仅配对模式）
│   └── comparison/            # 对比图（源域 vs 翻译 vs 目标域GT，配对模式）
├── cycle_ABA/                 # A→B→A 循环一致性
│   ├── original/              # 原始图像
│   ├── forward/               # 前向翻译（A→B）
│   ├── reconstructed/         # 循环重建（A→B→A）
│   └── comparison/            # 对比图
├── cycle_BAB/                 # B→A→B 循环一致性
│   ├── original/              # 原始图像
│   ├── forward/               # 前向翻译（B→A）
│   ├── reconstructed/         # 循环重建（B→A→B）
│   └── comparison/            # 对比图
├── identity_A/                # A 域恒等映射
│   └── identity_batch_*.png   # 恒等映射对比图
├── identity_B/                # B 域恒等映射
│   └── identity_batch_*.png   # 恒等映射对比图
├── A2B_c_visualization/       # A→B C 分量可视化
│   └── sample_*_c_components.png
├── B2A_c_visualization/       # B→A C 分量可视化
│   └── sample_*_c_components.png
└── evaluation_metrics.txt     # 评估报告
```

### 评估报告格式

`evaluation_metrics.txt` 包含定性评估和定量评估结果：

```
CycleDiff 模型评估报告 - 图像分量翻译能力
============================================================

定性评估:
  A2B 翻译:
    源域图像：evaluation/cyclediff/res/A2B_translation/source
    翻译图像：evaluation/cyclediff/res/A2B_translation/translated
    目标域GT：evaluation/cyclediff/res/A2B_translation/target_gt  （仅配对模式）
  循环一致性 ABA:
    原始图像：evaluation/cyclediff/res/cycle_ABA/original
    重建图像：evaluation/cyclediff/res/cycle_ABA/reconstructed

定量评估:
  ABA:
    pixel_l1_mean: 0.138938      # 像素级 L1 距离（越小越好）
    pixel_l1_std: 0.017911
    c_l1_mean: 0.396722          # C 空间 L1 距离（越小越好）
    c_l1_std: 0.039915
    original_path: ...
    reconstructed_path: ...
  identity_A:
    c_l1_mean: 0.074670          # 恒等映射 C 空间 L1
    c_l1_std: 0.038311
    c_cos_mean: 0.990317         # 恒等映射余弦相似度（越接近 1 越好）
    c_cos_std: 0.007322
  A2B_reconstruction:
    mse: 0.045678                # 均方误差
    psnr: 28.45 dB               # 峰值信噪比（越高越好）
    ssim: 0.8234                 # 结构相似性（越接近 1 越好）
    ms_ssim: 0.8567              # 多尺度结构相似性
    reference: target_gt         # 对比对象：target_gt（配对）或 source（非配对）
  A2B_c_metrics:
    c_l1_mean: 0.123456          # C 空间翻译 L1 距离
    c_l2_mean: 0.087654          # C 空间翻译 L2 距离
    c_cos_mean: 0.9876           # C 空间翻译余弦相似度
    reference: 目标域GT          # 对比对象：目标域GT（配对）或 源域（非配对）
  A2B_lpips:
    mean: 0.234567               # LPIPS（越低越好）
    std: 0.045678
    reference: 目标域GT          # 对比对象：目标域GT（配对）或 源域（非配对）

============================================================
评估完成!
```

### 终端输出示例

```
============================================================
CycleDiff 模型评估 - 图像分量翻译能力
============================================================

1. 加载配置文件：configs/maps/translation_C_disc_timestep_ode_2.yaml
2. 加载 CycleDiff 模型：results/maps/train_cycle_C_disc_G_blocks_12_maps_rsi2map/model-10.pt
   从 results/maps/train_cycle_C_disc_G_blocks_12_maps_rsi2map/model-10.pt 加载权重...
   使用 EMA 权重
   scale_factor: model1=0.165, model2=0.165
3. 加载数据集
   使用测试集进行评估
   A 域数据集大小：100
   B 域数据集大小：100
4. 创建结果保存目录：evaluation/cyclediff/res

============================================================
评估 1：图像分量翻译质量
============================================================

============================================================
评估 A→B 方向（RSI→Map）图像分量翻译质量
============================================================
  A→B 已处理 4/50 个样本
✓ A→B 翻译结果已保存到：evaluation/cyclediff/res/A2B_translation

============================================================
评估 2：循环一致性
============================================================

============================================================
评估 A→B→A 循环一致性
============================================================
  A→B→A 已处理 4/50 个样本
✓ A→B→A 循环一致性 - 像素 L1: 0.138938, C 空间 L1: 0.396722

============================================================
评估 3：恒等映射
============================================================

============================================================
评估恒等映射：net_G_B 应保持 A 域 C 不变
============================================================
  A 域恒等映射 已处理 4/50 个样本
✓ A 域恒等映射 - C 空间 L1: 0.074670, 余弦相似度：0.9903

============================================================
评估 4：C 分量逐时间步可视化
============================================================

============================================================
可视化 A→B 方向 C 分量逐时间步翻译
============================================================
  ✓ C 分量可视化已保存到：evaluation/cyclediff/res/A2B_c_visualization

============================================================
定量评估
============================================================

正在计算 A→B 翻译重建指标...
  MSE: 0.045678, PSNR: 28.45 dB, SSIM: 0.8234, MS-SSIM: 0.8567

正在计算 A→B 方向 C 空间翻译指标...
  A→B C 空间指标 - L1: 0.123456, L2: 0.087654, 余弦相似度：0.9876

正在计算 A→B 方向 LPIPS...
  A→B LPIPS: 0.234567 ± 0.045678

============================================================
评估总结
============================================================
✓ 循环一致性 ABA：良好 (像素 L1 = 0.138938)
  C 空间 L1: 0.396722
✓ 恒等映射 identity_A：优秀 (余弦相似度 = 0.9903)
✓ 翻译重建 A2B_reconstruction：良好 (PSNR = 28.45 dB)

✓ 评估完成！
============================================================
```

## 评估指标解读

### 循环一致性

- **像素 L1 < 0.1**：优秀，循环重建质量很高
- **像素 L1 0.1-0.2**：良好，循环一致性较好
- **像素 L1 > 0.2**：需改进，存在信息丢失

### 恒等映射

- **余弦相似度 > 0.95**：优秀，Generator 能很好地保持自身域特征
- **余弦相似度 0.90-0.95**：良好
- **余弦相似度 < 0.90**：需改进，可能存在模式坍塌

### 翻译重建质量

- **PSNR > 30 dB**：优秀
- **PSNR 25-30 dB**：良好
- **PSNR < 25 dB**：需改进
- **SSIM > 0.95**：优秀
- **SSIM 0.90-0.95**：良好
- **SSIM < 0.90**：需改进

### C 空间指标

- **C 空间余弦相似度 > 0.98**：C 分量翻译保真度很高
- **C 空间余弦相似度 0.95-0.98**：较好
- **C 空间余弦相似度 < 0.95**：C 分量翻译存在较大失真

## 常见问题

### Q1: 如何评估不同训练阶段的模型？

```bash
# 评估 model-2.pt
python evaluation/cyclediff/evaluate_cyclediff.py \
  --ckpt results/maps/train_cycle_C_disc_G_blocks_12_maps_rsi2map/model-2.pt \
  --save_dir evaluation/cyclediff/res_model2 \
  --cal_metrics \
  --use_test_set
```

### Q2: 如何仅评估单个方向？

```bash
# 仅评估 RSI→Map
python evaluation/cyclediff/evaluate_cyclediff.py \
  --direction A2B \
  --cal_metrics \
  --use_test_set

# 仅评估 Map→RSI
python evaluation/cyclediff/evaluate_cyclediff.py \
  --direction B2A \
  --cal_metrics \
  --use_test_set
```

### Q3: 如何快速测试（不计算定量指标）？

```bash
python evaluation/cyclediff/evaluate_cyclediff.py \
  --num_samples 10 \
  --use_test_set
```

### Q4: 如何评估训练集上的表现？

```bash
# 不使用 --use_test_set 参数即可
python evaluation/cyclediff/evaluate_cyclediff.py \
  --num_samples 20 \
  --cal_metrics
```

### Q5: LPIPS 计算失败怎么办？

LPIPS 需要 `taming` 模块的支持。如果遇到 LPIPS 计算失败，评估脚本会继续执行其他指标的计算。如需修复 LPIPS，请确保：

```bash
# 检查 taming 模块是否存在
ls -la /root/autodl-tmp/CycleDiff/taming/modules/losses/lpips.py
```

### Q6: 配对评估模式需要什么数据格式？

配对模式要求源域和目标域图像**文件名完全一致**，按文件名自动配对。例如：

```
data/test/class_RSI/    data/test/class_Map/
  ├── img_001.png         ├── img_001.png
  ├── img_002.png         ├── img_002.png
  └── img_003.png         └── img_003.png
```

脚本会自动查找两个目录中同名文件进行配对。如果配对数据集为空，会输出警告信息。

### Q7: 为什么非配对模式下翻译质量指标（MSE/PSNR/SSIM）没有意义？

图像翻译任务的目标是将源域图像转换为目标域风格（如 RSI→Map），翻译结果**本应不同于源域**。因此：

- **非配对模式**：翻译结果 vs 源域图像 → 数值高不代表翻译好，数值低也不代表翻译差
- **配对模式**：翻译结果 vs 目标域 GT → 数值直接反映翻译准确度

> 建议：最终评估报告使用配对模式，非配对模式仅用于循环一致性和身份映射验证。

### Q8: 如何同时使用 EMA 权重和配对模式？

```bash
python evaluation/cyclediff/evaluate_cyclediff.py \
  --use_ema \
  --paired \
  --cal_metrics \
  --use_test_set
```

## 技术细节

### C 分量的物理意义

在 CycleDiff 的前向扩散过程中：

```
x_t = x_0 + C*t + t*ε
```

其中：

- `x_0`：原始图像（潜空间）
- `C = -x_0`：图像分量（确定性漂移项）
- `ε`：噪声分量
- `t`：时间步（0 到 1）

Generator 的作用是学习域间映射：`C_T = net_G(C_S, t)`，在不同时间步 t 下将源域的 C\_S 映射为目标域的 C\_T。

### 评估流程

1. **C 列表提取**：使用 `reverse_q_sample_c_list_concat` 沿反向扩散路径分解图像，得到各时间步的 C 列表
2. **C 翻译**：Generator 逐时间步翻译 C 分量
3. **目标域采样**：使用 `sample_from_c_list` 从翻译后的 C 列表重建图像
4. **指标计算**：对比原始图像与翻译/循环重建图像

### EMA 权重

默认使用 EMA（指数移动平均）权重进行评估，通常能提供更稳定的结果。如需使用原始权重：

```bash
python evaluation/cyclediff/evaluate_cyclediff.py --use_ema
```

## 参考资料

- [CycleDiff 训练脚本](../../train_uncond_ldm_cycle.py)
- [CycleDiff 推理脚本](../../translation_uncond_ldm_cycle.py)
- [LDM 评估脚本](../ldm/evaluate_ldm.py)
- [CycleDiff 配置文件](../../configs/maps/translation_C_disc_timestep_ode_2.yaml)

## 更新日志

- **2024-XX-XX**：初始版本，包含四大评估模块和完整的定量指标
- **2024-XX-XX**：
  - 新增配对评估模式（`--paired`），支持翻译结果与目标域 GT 对比
  - 修复 `--use_ema` 参数默认值（改为 `False`，可正常禁用）
  - 修复循环一致性评估中 `fwd_img` 值域不匹配的问题
  - 修复训练集评估时随机水平翻转导致结果不确定的问题
  - 更新所有定量指标输出，添加 `reference` 字段标明对比对象

