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
