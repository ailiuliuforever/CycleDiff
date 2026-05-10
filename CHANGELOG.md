# CycleDiff 修改日志
生成时间: 2026-05-10 15:33:22

## 修改目的
解决 C-space 翻译中 RSI→Map 方向的 C 分量退化为纯色块的问题。

## 问题诊断
- Map 域图像空间结构稀疏（大面积均匀区域），C 分量天然趋近常数
- Generator 在梯度下降中倾向于"抹平"难以建模的边缘结构
- 纯色块在 L1/L2 损失下误差小，但视觉结构完全丢失
- 训练中 perceptual loss 从 0.84 爆炸到 13.13，cycle_ABA 从 10.24 升到 16.47

## 解决方案
添加 C 分量梯度损失（边缘加权版），在梯度空间惩罚翻译后 C 分量的结构退化。

### 原理
- 纯色块 → 空间梯度 ≈ 0
- 真 Map → 边缘处梯度 > 0
- 梯度损失直接惩罚"抹平"行为，尤其加重边缘区域权重

## 修改文件

### 1. util/c_gradient_loss.py (新建)
- 函数: c_gradient_loss_weighted(c_translated, c_target, edge_weight=10.0)
- 计算翻译后 C 分量与真实 C 分量在空间梯度上的 L1 差异
- 边缘区域（target 梯度大的位置）权重放大 edge_weight 倍
- .detach() 防止权重参与梯度计算

### 2. configs/maps/translation_C_disc_timestep_ode_2.yaml
新增参数:
  c_gradient_weight: 10.0         # C分量梯度损失权重
  c_gradient_edge_boost: 10.0     # 边缘区域加权倍数

### 3. train_uncond_ldm_cycle_swanlab.py
- 第18行: 添加 import c_gradient_loss_weighted
- 第610-615行: 在 fake_C_T 翻译后计算 loss_c_grad
- 第642行: 总损失中添加 + loss_c_grad
- 第650行: loss_dict 中添加日志记录

## 训练集成点
```python
# L604: fake_C_T = self.net_G_A(input_C_S, t)  [RSI→Map C分量翻译]
# ↓ 新增 ↓
loss_c_grad = torch.tensor(0.0, device=fake_C_T.device)
if hasattr(self.cfg.trainer, "c_gradient_weight"):
    loss_c_grad = c_gradient_loss_weighted(
        fake_C_T, input_C_T,
        edge_weight=self.cfg.trainer.get('c_gradient_edge_boost', 10.0)
    ) * self.cfg.trainer.c_gradient_weight
# L605: recon_C_S = self.net_G_B(fake_C_T, t)  [Map→RSI 回译]
```

## 调参建议

| 阶段 | c_gradient_weight | 理由 |
|------|------------------|------|
| 初始 (0-10k步) | 3.0 | 温和引入，观察是否导致 loss 爆炸 |
| 中期 (10k-50k步) | 10.0 | 加强结构约束 |
| 后期 (50k+步) | 5.0 | 略降，交给对抗损失 |

监测指标:
- train/loss_c_grad 应逐步下降
- train/loss_perceptual 不应爆炸
- train/loss_cycle_ABA 应趋于稳定

## 验证结果
```
1. c_gradient_loss_weighted imported OK
2. c_gradient_weight: 10.0 ✓
3. training script syntax OK ✓
4. functional test: loss = 42.05 ✓
```

## 后续计划
- 同时调整损失平衡 (cycle vs adversarial)
- 考虑降低 cycle 权重以缓解梯度冲突
- 降低 trans_net_lr 以减少训练震荡
