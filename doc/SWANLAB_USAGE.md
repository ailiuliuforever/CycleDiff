# SwanLab 实验追踪使用说明

## 概述
`train_vae_swanlab.py` 是一个集成了 SwanLab 实验追踪功能的 VAE 训练脚本，基于原有的 `train_vae.py` 创建。该脚本提供完整的训练过程追踪、实验结果可视化以及训练日志的系统化保存功能。

## 主要功能

### 1. 实验初始化
- 自动从配置文件路径提取实验名称
- 创建 SwanLab 项目 "CycleDiff-RSI2Map"
- 记录完整的超参数配置

### 2. 训练指标追踪
实时记录以下训练指标到 SwanLab：
- `Learning_Rate`: 学习率
- `total_loss`: 总损失
- `rec_loss`: 重建损失
- `kl_loss`: KL 散度损失
- `d_weight`: 判别器权重
- `disc_factor`: 判别器因子
- `g_loss`: 生成器损失
- `disc_loss`: 判别器损失
- `logits_real`: 真实样本判别得分
- `logits_fake`: 生成样本判别得分

### 3. 图像可视化
- 定期保存重建的图像样本
- 自动上传图像到 SwanLab 仪表板
- 包含步数和里程碑信息

### 4. 超参数配置
自动记录以下超参数：
- `model_class`: 模型类别
- `embed_dim`: 嵌入维度
- `batch_size`: 批处理大小
- `learning_rate`: 学习率
- `min_lr`: 最小学习率
- `train_num_steps`: 训练步数
- `image_size`: 图像尺寸
- `gradient_accumulate_every`: 梯度累积步数
- `save_and_sample_every`: 保存和采样频率
- `amp`: 是否使用自动混合精度
- `fp16`: 是否使用 FP16

## 使用方法

### 基本用法
```bash
python train_vae_swanlab.py --cfg configs/rsi2map/rsi_ae_kl_256x256_d4.yaml
```

### 针对不同数据集的训练

#### 1. 训练 RSI 自编码器
```bash
python train_vae_swanlab.py --cfg configs/rsi2map/rsi_ae_kl_256x256_d4.yaml
```

#### 2. 训练 Map 自编码器
```bash
python train_vae_swanlab.py --cfg configs/rsi2map/map_ae_kl_256x256_d4.yaml
```

## 查看实验结果

### 本地查看
训练完成后，可以使用以下命令在本地查看 SwanLab 实验看板：
```bash
swanlab watch -l ./swanlog
```

### 云端查看
如果配置了 SwanLab 云端同步，可以访问 SwanLab 官网查看实验结果。

## 与 TensorBoard 的兼容性

本脚本同时保留了 TensorBoard 功能，训练日志会同时写入：
- SwanLab（云端/本地看板）
- TensorBoard（`results_folder` 目录下的 `runs` 文件夹）

您可以选择使用任意一种或两种方式查看训练进度。

## 断点续训

脚本支持断点续训功能。如果训练中断，再次运行相同的命令会自动加载最近的检查点继续训练。

## 注意事项

1. **确保 SwanLab 已安装**：
   ```bash
   pip install swanlab
   ```

2. **实验名称**：实验名称自动从配置文件路径提取，确保配置文件命名规范。

3. **图像记录**：图像样本在达到保存里程碑时自动记录到 SwanLab，如果记录失败会在日志中显示警告但不影响训练。

4. **数据同步**：训练结束时会自动调用 `swanlab.finish()` 确保所有数据同步完成。

## 示例输出

训练过程中，您会在终端看到类似以下的输出：
```
[Train Step] 100/50000: lr: 0.0001, total_loss: 1.234, rec_loss: 0.567, kl_loss: 0.123, ...
```

同时在 SwanLab 仪表板中可以查看：
- 实时更新的损失曲线
- 学习率变化曲线
- 重建的图像样本
- 完整的超参数配置

## 故障排除

### 问题：SwanLab 初始化失败
**解决方案**：检查网络连接或尝试离线模式运行。

### 问题：图像记录失败
**解决方案**：检查图像路径是否正确，确保有写入权限。

### 问题：训练中断后无法恢复
**解决方案**：检查 `results_folder` 下是否存在检查点文件，确认 `resume_milestone` 配置正确。

---

## 扩散模型（LDM）训练

### 概述
`train_ldm_swanlab.py` 是一个集成了 SwanLab 实验追踪功能的**潜在扩散模型（Latent Diffusion Model, LDM）**训练脚本。该脚本用于在 VAE 的潜在空间中训练扩散模型，实现高质量的图像生成和图像翻译任务。

### 主要功能

#### 1. 训练指标追踪
实时记录以下训练指标到 SwanLab：
- `Learning_Rate`: 学习率
- `total_loss`: 总损失
- `loss_simple`: 简化损失（噪声预测误差）
- `loss_vlb`: 变分下界损失（分布匹配误差）

#### 2. 图像可视化
- 定期保存生成的图像样本
- 自动上传图像到 SwanLab 仪表板
- 包含步数、里程碑和采样步数信息

#### 3. 超参数配置
自动记录以下超参数：
- `model_class`: 模型类别（如 LatentDiffusion）
- `batch_size`: 批处理大小
- `learning_rate`: 学习率
- `min_lr`: 最小学习率
- `train_num_steps`: 训练步数
- `image_size`: 图像尺寸
- `gradient_accumulate_every`: 梯度累积步数
- `save_and_sample_every`: 保存和采样频率
- `amp`: 是否使用自动混合精度
- `fp16`: 是否使用 FP16
- `sampling_timesteps`: 采样步数（如 DDIM 步数）

### 使用方法

#### 基本用法
```bash
python train_ldm_swanlab.py --cfg configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml
```

#### 针对不同数据集的训练

##### 1. 训练 RSI 潜在扩散模型
```bash
python train_ldm_swanlab.py --cfg configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml
```

##### 2. 训练 Map 潜在扩散模型
```bash
python train_ldm_swanlab.py --cfg configs/rsi2map/map_ddm_const4_ldm_unet6_114_ode_2.yaml
```

#### 使用 Accelerate 进行分布式训练
```bash
# 单卡训练
accelerate launch train_ldm_swanlab.py --cfg configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml

# 多卡训练（根据实际 GPU 数量调整）
accelerate launch --multi_gpu --num_processes=4 train_ldm_swanlab.py --cfg configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml
```

### 训练流程说明

#### 1. 准备工作
在训练 LDM 之前，需要：
- ✅ 完成 VAE 训练（VAE 是 LDM 的基础）
- ✅ 修改配置文件中的 VAE 权重路径
- ✅ 确认数据集路径配置正确

#### 2. 配置文件示例
```yaml
# configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml
model:
  class_name: ddm.latent_diffusion.LatentDiffusion
  first_stage:
    class_name: ddm.encoder_decoder.AutoencoderKL
    embed_dim: 3
    ckpt_path: "results/rsi_ae_kl_256x256_d4/model-10.pt"  # ← VAE 权重
  unet:
    class_name: ddm.unet.UNet
    # UNet 配置...
  sampling_timesteps: 100  # DDIM 采样步数

data:
  batch_size: 16
  num_workers: 4

trainer:
  lr: !!float 1e-4
  min_lr: !!float 1e-6
  train_num_steps: 100000
  save_and_sample_every: 5000
  log_freq: 200
```

#### 3. 训练过程
```bash
# 启动训练
python train_ldm_swanlab.py --cfg configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml

# 终端输出示例
[Train Step] 100/100000: lr: 0.0001, total_loss: 0.856, loss_simple: 0.723, loss_vlb: 0.133
[Train Step] 200/100000: lr: 0.0001, total_loss: 0.745, loss_simple: 0.634, loss_vlb: 0.111
...
```

### 示例输出

训练过程中，您会在终端看到类似以下的输出：
```
[Train Step] 1000/100000: lr: 0.000095, total_loss: 0.523, loss_simple: 0.445, loss_vlb: 0.078
```

同时在 SwanLab 仪表板中可以查看：
- 实时更新的损失曲线（total_loss, loss_simple, loss_vlb）
- 学习率变化曲线
- 生成的图像样本（每 5000 步保存一次）
- 完整的超参数配置

### 训练完成后

#### 1. 查看实验结果
```bash
# 本地查看 SwanLab
swanlab watch -l ./swanlog

# 或访问云端看板（如果配置了同步）
```

#### 2. 使用训练好的 LDM 模型
```bash
# 生成图像
python sample_ldm.py --cfg configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml \
                     --ckpt results/rsi_ddm_const4_ldm_unet6_114_ode_2/model-20.pt \
                     --num_samples 16
```

### 注意事项

1. **依赖 VAE 模型**：
   - LDM 训练依赖于预训练好的 VAE
   - 确保配置文件中的 `ckpt_path` 指向正确的 VAE 权重

2. **显存需求**：
   - LDM 训练比 VAE 需要更多显存
   - 建议使用至少 16GB 显存的 GPU
   - 可以开启 `amp: True` 使用混合精度训练节省显存

3. **训练时间**：
   - LDM 通常需要训练 50,000-100,000 步
   - 根据数据集大小和 GPU 性能，可能需要数小时到数天

4. **采样步数**：
   - `sampling_timesteps` 影响生成质量和速度
   - 推荐值：100-500（平衡质量和速度）
   - 训练时可以设置较小值（如 100），推理时可以增大

5. **断点续训**：
   - 训练中断后，再次运行相同命令会自动加载最近的检查点
   - 检查点保存在 `results_folder` 目录下

### 与 VAE 训练的对比

| 特性 | VAE 训练 | LDM 训练 |
|------|----------|----------|
| **脚本** | `train_vae_swanlab.py` | `train_ldm_swanlab.py` |
| **依赖** | 无 | 需要预训练 VAE |
| **训练指标** | rec_loss, kl_loss, d_weight | loss_simple, loss_vlb |
| **图像输出** | 重建图像 | 生成图像 |
| **训练步数** | 30,000-50,000 | 50,000-100,000 |
| **显存需求** | 中等（~8GB） | 较高（~16GB） |
| **训练时间** | 较短（数小时） | 较长（数小时到数天） |

### 完整训练流程示例

```bash
# 步骤 1：训练 VAE
python train_vae_swanlab.py --cfg configs/rsi2map/rsi_ae_kl_256x256_d4.yaml

# 步骤 2：等待 VAE 训练完成，记录模型路径
# 例如：results/rsi_ae_kl_256x256_d4/model-10.pt

# 步骤 3：修改 LDM 配置文件，更新 VAE 权重路径
# vim configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml
# 修改：ckpt_path: "results/rsi_ae_kl_256x256_d4/model-10.pt"

# 步骤 4：训练 LDM
python train_ldm_swanlab.py --cfg configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml

# 步骤 5：在 SwanLab 中查看训练进度
swanlab watch -l ./swanlog
```
