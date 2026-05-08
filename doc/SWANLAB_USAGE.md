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

---

## CycleDiff 循环一致性训练

### 概述
`train_uncond_ldm_cycle_swanlab.py` 是一个集成了 SwanLab 实验追踪功能的 **CycleDiff 循环一致性训练**脚本。该脚本用于训练两个潜在扩散模型（LDM）以及两个生成器（net_G_A/net_G_B）和两个判别器（net_D_A/net_D_B），实现无配对图像到图像的翻译任务。

### 主要功能

#### 1. 训练指标追踪
实时记录以下训练指标到 SwanLab：

**生成器指标：**
- `Generator/loss_gen_toal`: 生成器总损失
- `Generator/loss_idt`: Identity 损失
- `Generator/loss_G_adv_A`: 生成器对抗损失 A
- `Generator/loss_G_adv_B`: 生成器对抗损失 B
- `Generator/loss_cycle_ABA`: Cycle ABA 损失
- `Generator/loss_cycle_BAB`: Cycle BAB 损失
- `Generator/loss_ldm`: LDM 损失
- `Generator/loss_perceptual`: 感知损失

**判别器指标：**
- `Discriminator/loss_dis_total`: 判别器总损失
- `Discriminator/loss_D_A`: 判别器 A 损失
- `Discriminator/loss_D_B`: 判别器 B 损失
- `Discriminator/loss_mcl_A`: MCL 损失 A
- `Discriminator/loss_mcl_B`: MCL 损失 B
- `Discriminator/loss_ldm_D`: 判别器 LDM 损失

**学习率指标：**
- `Learning_Rate/d1`: 扩散模型1 学习率
- `Learning_Rate/d2`: 扩散模型2 学习率
- `Learning_Rate/G`: 生成器学习率
- `Learning_Rate/D`: 判别器学习率

**其他指标：**
- `total_loss`: 总损失

#### 2. 图像可视化
定期保存并上传以下图像到 SwanLab 仪表板：
- `samples/source_A`: 源域 A 图像
- `samples/source_B`: 源域 B 图像
- `samples/model_A`: 模型 A 生成样本
- `samples/model_B`: 模型 B 生成样本
- `samples/translation_A2B`: A→B 翻译结果
- `samples/translation_B2A`: B→A 翻译结果

#### 3. 超参数配置
自动记录以下超参数：
- `model1_class` / `model2_class`: 模型类别
- `batch_size`: 批处理大小
- `train_lr`: 扩散模型学习率
- `trans_net_lr`: 翻译网络学习率
- `min_lr`: 最小学习率
- `train_num_steps`: 训练步数
- `image_size`: 图像尺寸
- `gradient_accumulate_every`: 梯度累积步数
- `save_every`: 保存频率
- `sample_every`: 采样频率
- `amp`: 是否使用自动混合精度
- `fp16`: 是否使用 FP16
- `sampling_timesteps`: 采样步数
- `cycle_weight`: 循环一致性权重
- `idt_weight`: Identity 损失权重
- `perceptual_weight`: 感知损失权重
- `mcl_weight`: MCL 损失权重

### 使用方法

#### 基本用法
```bash
python train_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_C_disc_timestep_ode_2.yaml
```

#### 使用 Accelerate 进行分布式训练
```bash
# 单卡训练
accelerate launch train_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_C_disc_timestep_ode_2.yaml

# 多卡训练（根据实际 GPU 数量调整）
accelerate launch --multi_gpu --num_processes=4 train_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_C_disc_timestep_ode_2.yaml
```

### 训练流程说明

#### 1. 准备工作
在训练 CycleDiff 之前，需要：
- ✅ 完成两个域的 VAE 训练
- ✅ 完成两个域的 LDM 训练
- ✅ 修改配置文件中的 VAE 和 LDM 权重路径
- ✅ 确认数据集路径配置正确

#### 2. 配置文件示例
```yaml
# configs/rsi2map/translation_C_disc_timestep_ode_2.yaml
model1:
  class_name: ddm.latent_diffusion.LatentDiffusion
  first_stage:
    class_name: ddm.encoder_decoder.AutoencoderKL
    embed_dim: 3
    ckpt_path: "results/rsi_ae_kl_256x256_d4/model-10.pt"
  unet:
    class_name: ddm.unet.UNet
    # UNet 配置...
  sampling_timesteps: 100

model2:
  class_name: ddm.latent_diffusion.LatentDiffusion
  first_stage:
    class_name: ddm.encoder_decoder.AutoencoderKL
    embed_dim: 3
    ckpt_path: "results/map_ae_kl_256x256_d4/model-10.pt"
  unet:
    class_name: ddm.unet.UNet
    # UNet 配置...
  sampling_timesteps: 100

net_G:
  class_name: ddm.cycle_generator_2.TransNet
  # 生成器配置...

net_D:
  class_name: ddm.cycle_discriminator.Discriminator
  # 判别器配置...

data:
  class_name: ddm.data.YourDataset
  batch_size: 16
  num_workers: 4

trainer:
  lr: !!float 1e-4
  trans_net_lr: !!float 1e-5
  min_lr: !!float 1e-6
  train_num_steps: 100000
  save_every: 5000
  sample_every: 5000
  log_freq: 200
  cycle_weight: 10.0
  idt_weight: 5.0
  perceptual_weight: 1.0
  mcl_weight: 1.0
```

#### 3. 训练过程
```bash
# 启动训练
python train_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_C_disc_timestep_ode_2.yaml

# 终端输出示例
[Train Step] 100/100000: lr_d1: 0.0001, lr_d2: 0.0001, lr_G: 0.00001, lr_D: 0.00001, loss_gen_toal: 2.345, loss_idt: 0.567, loss_G_adv_A: 0.234, ...
[Train Step] 200/100000: lr_d1: 0.0001, lr_d2: 0.0001, lr_G: 0.00001, lr_D: 0.00001, loss_gen_toal: 1.876, loss_idt: 0.456, loss_G_adv_A: 0.198, ...
...
```

### 示例输出

训练过程中，您会在终端看到类似以下的输出：
```
[Train Step] 1000/100000: lr_d1: 0.000095, lr_d2: 0.000095, lr_G: 0.0000095, lr_D: 0.0000095, loss_gen_toal: 1.234, loss_idt: 0.345, loss_G_adv_A: 0.156, loss_G_adv_B: 0.167, loss_cycle_ABA: 0.234, loss_cycle_BAB: 0.245, loss_ldm: 0.456, loss_perceptual: 0.123, loss_dis_total: 0.567, loss_D_A: 0.234, loss_D_B: 0.245, loss_mcl_A: 0.123, loss_mcl_B: 0.134, loss_ldm_D: 0.456
```

同时在 SwanLab 仪表板中可以查看：
- 实时更新的生成器损失曲线（loss_gen_toal, loss_idt, loss_G_adv_A, loss_G_adv_B, loss_cycle_ABA, loss_cycle_BAB, loss_ldm, loss_perceptual）
- 实时更新的判别器损失曲线（loss_dis_total, loss_D_A, loss_D_B, loss_mcl_A, loss_mcl_B, loss_ldm_D）
- 4 个优化器的学习率变化曲线
- 总损失变化曲线
- 生成的图像样本（source_A, source_B, model_A, model_B, translation_A2B, translation_B2A）
- 完整的超参数配置

### 训练完成后

#### 1. 查看实验结果
```bash
# 本地查看 SwanLab
swanlab watch -l ./swanlog

# 或访问云端看板（如果配置了同步）
```

#### 2. 使用训练好的 CycleDiff 模型
```bash
# 图像翻译
python translation_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_C_disc_timestep_ode_2.yaml
```

### 注意事项

1. **依赖预训练模型**：
   - CycleDiff 训练依赖于预训练好的 VAE 和 LDM
   - 确保配置文件中的 `ckpt_path` 指向正确的权重

2. **显存需求**：
   - CycleDiff 训练需要同时加载两个 LDM、两个生成器和两个判别器
   - 建议使用至少 24GB 显存的 GPU
   - 可以开启 `amp: True` 使用混合精度训练节省显存

3. **训练时间**：
   - CycleDiff 通常需要训练 100,000-200,000 步
   - 根据数据集大小和 GPU 性能，可能需要数天

4. **损失平衡**：
   - `cycle_weight`: 控制循环一致性损失的权重（默认 10.0）
   - `idt_weight`: 控制 Identity 损失的权重（默认 5.0）
   - `perceptual_weight`: 控制感知损失的权重（默认 1.0）
   - `mcl_weight`: 控制 MCL 损失的权重（默认 1.0）
   - 根据具体任务调整这些权重以获得最佳效果

5. **断点续训**：
   - 训练中断后，再次运行相同命令会自动加载最近的检查点
   - 检查点保存在 `results_folder` 目录下

### 与其他训练的对比

| 特性 | VAE 训练 | LDM 训练 | CycleDiff 训练 |
|------|----------|----------|----------------|
| **脚本** | `train_vae_swanlab.py` | `train_ldm_swanlab.py` | `train_uncond_ldm_cycle_swanlab.py` |
| **依赖** | 无 | 需要预训练 VAE | 需要预训练 VAE 和 LDM |
| **训练指标** | rec_loss, kl_loss, d_weight | loss_simple, loss_vlb | loss_gen_toal, loss_idt, loss_cycle, loss_D, loss_mcl |
| **图像输出** | 重建图像 | 生成图像 | 源图像、生成图像、翻译图像 |
| **训练步数** | 30,000-50,000 | 50,000-100,000 | 100,000-200,000 |
| **显存需求** | 中等（~8GB） | 较高（~16GB） | 高（~24GB） |
| **训练时间** | 较短（数小时） | 较长（数小时到数天） | 长（数天） |

### 完整训练流程示例

```bash
# 步骤 1：训练源域 VAE
python train_vae_swanlab.py --cfg configs/rsi2map/rsi_ae_kl_256x256_d4.yaml

# 步骤 2：训练目标域 VAE
python train_vae_swanlab.py --cfg configs/rsi2map/map_ae_kl_256x256_d4.yaml

# 步骤 3：训练源域 LDM
python train_ldm_swanlab.py --cfg configs/rsi2map/rsi_ddm_const4_ldm_unet6_114_ode_2.yaml

# 步骤 4：训练目标域 LDM
python train_ldm_swanlab.py --cfg configs/rsi2map/map_ddm_const4_ldm_unet6_114_ode_2.yaml

# 步骤 5：修改 CycleDiff 配置文件，更新权重路径
# vim configs/rsi2map/translation_C_disc_timestep_ode_2.yaml

# 步骤 6：训练 CycleDiff
python train_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_C_disc_timestep_ode_2.yaml

# 步骤 7：在 SwanLab 中查看训练进度
swanlab watch -l ./swanlog
```

---

## 图像翻译推理（Translation）

### 概述
`translation_uncond_ldm_cycle_swanlab.py` 是一个集成了 SwanLab 实验追踪功能的**图像翻译推理**脚本。该脚本用于加载训练好的 CycleDiff 模型，对整个测试集进行图像翻译，并支持计算多种评估指标（FID、L2、MSE、PSNR、SSIM）。

### 主要功能

#### 1. 推理进度追踪
实时记录以下推理进度指标到 SwanLab：
- `progress/batch_index`: 当前批次索引
- `progress/samples_processed`: 已处理的样本数量
- `progress/batch_time`: 每批次处理时间
- `progress/total_time`: 总处理时间

#### 2. 翻译样本可视化
定期保存并上传翻译结果到 SwanLab 仪表板：
- `translation_samples/source`: 源域输入图像
- `translation_samples/translated`: 翻译后的目标域图像

#### 3. 评估指标计算
支持计算以下评估指标（需在配置中开启 `cal_metrics: True`）：
- `evaluation/FID`:  Fréchet Inception Distance
- `evaluation/L2`: L2 距离
- `evaluation/MSE`: 均方误差
- `evaluation/PSNR`: 峰值信噪比
- `evaluation/SSIM`: 结构相似性指数

#### 4. 支持的任务类型
支持多种图像翻译任务：
- `cat2dog` / `dog2cat`: 猫狗图像互转
- `wild2dog` / `dog2wild`: 野生动物与家养狗互转
- `male2female` / `female2male`: 性别转换
- `sem2rgb` / `rgb2sem`: 语义图与RGB互转
- `edge2rgb` / `rgb2edge`: 边缘图与RGB互转
- `depth2rgb` / `rgb2depth`: 深度图与RGB互转
- `summer2winter` / `winter2summer`: 季节转换
- `horse2zebra` / `zebra2horse`: 马与斑马互转
- `young2old` / `old2young`: 年龄转换
- `map2satellite` / `satellite2map`: 地图与卫星图互转
- `label2cityscape` / `cityscape2label`: 城市景观标签与图像互转
- `rsi2map` / `map2rsi`: 遥感图像与地图互转

### 使用方法

#### 基本用法
```bash
python translation_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_C_disc_timestep_ode_2.yaml
```

#### 配置文件示例
```yaml
# configs/rsi2map/translation_inference.yaml
sampler:
  task: "rsi2map"                    # 翻译任务类型
  ckpt_path: "results/cyclediff/model-final.pt"  # CycleDiff 训练好的模型权重
  save_folder: "results/translation/rsi2map"     # 翻译结果保存路径
  batch_size: 16                     # 推理批次大小
  use_ema: True                      # 是否使用 EMA 权重
  log_image_freq: 10                 # 每多少批次记录一次图像到 SwanLab
  cal_metrics: True                  # 是否计算评估指标
  source_gt_path: "data/rsi/test"    # 源域 Ground Truth 路径（用于计算指标）
  target_gt_path: "data/map/test"    # 目标域 Ground Truth 路径（用于计算FID）

data_test:
  class_name: ddm.data.RSIDataset
  data_root: "data/rsi/test"
  image_size: 256
  num_workers: 4

data_test2:
  class_name: ddm.data.MapDataset
  data_root: "data/map/test"
  image_size: 256
  num_workers: 4
```

#### 使用 Accelerate 进行分布式推理
```bash
# 单卡推理
accelerate launch translation_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_inference.yaml

# 多卡推理（根据实际 GPU 数量调整）
accelerate launch --multi_gpu --num_processes=4 translation_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_inference.yaml
```

### 推理流程说明

#### 1. 准备工作
在进行图像翻译之前，需要：
- ✅ 完成 CycleDiff 训练
- ✅ 准备测试集数据
- ✅ 修改配置文件中的模型权重路径和数据集路径

#### 2. 推理过程
```bash
# 启动翻译
python translation_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_inference.yaml

# 终端输出示例
sampling complete
```

翻译过程中，SwanLab 会实时显示：
- 当前处理的批次进度
- 已处理的样本数量
- 每批次处理时间
- 源图像和翻译结果的预览

#### 3. 查看翻译结果

##### 本地查看 SwanLab
```bash
swanlab watch -l ./swanlog
```

##### 查看保存的图像
翻译结果会保存在配置文件中指定的 `save_folder` 目录下，每个图像以 PNG 格式保存。

### 示例输出

推理过程中，您会在终端看到类似以下的输出：
```
sampling complete
Calculating metrics for task: rsi2map
Translation path: results/translation/rsi2map
Source GT path: data/rsi/test
Target GT path: data/map/test
FID: 45.23
L2: 0.1567
MSE: 0.0245
PSNR: 16.12
SSIM: 0.7234
Evaluation metrics logged to SwanLab successfully.
```

同时在 SwanLab 仪表板中可以查看：
- 实时更新的推理进度（批次索引、已处理样本数、处理时间）
- 源图像与翻译结果的对比预览
- 完整的评估指标（FID、L2、MSE、PSNR、SSIM）
- 完整的超参数配置

### 注意事项

1. **依赖预训练模型**：
   - 翻译脚本依赖于训练好的 CycleDiff 模型
   - 确保配置文件中的 `ckpt_path` 指向正确的权重文件

2. **显存需求**：
   - 推理时需要同时加载两个 LDM 和两个生成器
   - 建议根据 GPU 显存调整 `batch_size`

3. **推理速度**：
   - 脚本会遍历整个测试集
   - 推理时间取决于测试集大小、图像尺寸和 GPU 性能

4. **指标计算**：
   - 计算评估指标需要 Ground Truth 数据
   - 确保 `source_gt_path` 和 `target_gt_path` 配置正确
   - FID 计算需要 `target_gt_path`，其他指标需要 `source_gt_path`

5. **结果保存**：
   - 翻译结果默认保存为 PNG 格式
   - 文件名与输入图像保持一致

### 与其他脚本的对比

| 特性 | VAE 训练 | LDM 训练 | CycleDiff 训练 | 图像翻译推理 |
|------|----------|----------|----------------|--------------|
| **脚本** | `train_vae_swanlab.py` | `train_ldm_swanlab.py` | `train_uncond_ldm_cycle_swanlab.py` | `translation_uncond_ldm_cycle_swanlab.py` |
| **阶段** | 预训练 | 预训练 | 训练 | 推理 |
| **依赖** | 无 | 需要预训练 VAE | 需要预训练 VAE 和 LDM | 需要训练好的 CycleDiff |
| **指标** | rec_loss, kl_loss | loss_simple, loss_vlb | loss_gen, loss_D | FID, L2, MSE, PSNR, SSIM |
| **输出** | 重建图像 | 生成图像 | 训练样本 | 翻译图像 |

### 完整使用流程示例

```bash
# 步骤 1-6：完成 CycleDiff 训练（见上文）

# 步骤 7：准备翻译配置文件
# vim configs/rsi2map/translation_inference.yaml
# 配置：
#   - sampler.ckpt_path: CycleDiff 训练好的模型路径
#   - sampler.task: 翻译任务类型
#   - sampler.save_folder: 结果保存路径
#   - data_test / data_test2: 测试集路径

# 步骤 8：执行图像翻译
python translation_uncond_ldm_cycle_swanlab.py --cfg configs/rsi2map/translation_inference.yaml

# 步骤 9：在 SwanLab 中查看翻译结果和评估指标
swanlab watch -l ./swanlog
```
