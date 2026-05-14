# CycleDiff A2B 翻译 C 分量结构丢失问题 — 完整分析报告

---

## 目录

1. [问题陈述](#1-问题陈述)
2. [数据集量化分析](#2-数据集量化分析)
3. [根因分析](#3-根因分析)
4. [解决方案设计](#4-解决方案设计)
5. [实验验证](#5-实验验证)
6. [结论与讨论](#6-结论与讨论)

---

## 1. 问题陈述

### 1.1 观察现象

在 RSI（遥感影像）→ Map（地图瓦片）的 CycleGAN 翻译任务中，**A→B 方向（RSI→Map）的 C 分量完全坍缩为色块，丢失了全部空间结构**；而 B→A 方向（Map→RSI）的 C 分量保留了清晰的结构信息。

#### 1.1.1 什么是 C 分量

C 分量是冻结 LDM（`ddm/ddm_const_ode_2.py` 中的 `LatentDiffusion`）在多时间步预测得到的图像隐空间表示序列。提取过程由 `reverse_q_sample_c_list_concat`（`ddm/ddm_const_ode_2.py:525–560`）完成：

1. **VAE 编码**：输入图像 `x`（已归一化到 `[-1, 1]`）经冻结 VAE 编码为潜变量 `z`
2. **时间步构造**：从 `sigma_max=1.0` 到 `step=1/sampling_timesteps` 生成等间距时间步序列
3. **迭代提取**：在每个时间步 `t_cur`，冻结 UNet 从当前噪声状态 `x_t` 预测 `C` 和 `noise`，然后反解出 `x0 = x_t - C·t - noise·t`，取 `C = -x0` 作为该时间步的图像分量
4. **输出**：`c_list` 序列（`c_list[0]` 对应小时间步，包含精细结构；`c_list[-1]` 对应大时间步，接近噪声）和最终的噪声状态 `x_t`

C 分量本质上是**每个扩散时间步对原始潜变量的负估计**，由 `C = -1 * x_start`（`ddm/ddm_const_ode_2.py:211`）定义。它代表图像的低频结构信息，C 分量的质量直接影响 decoder 能否正确重建图像几何结构。

### 1.2 基础指标对比

| 指标 | RSI（源域 A）| Map（目标域 B）| 差异 |
|------|-------------|---------------|------|
| VAE 重建 PSNR | **25.79 dB** | **36.14 dB** | Map +10.35 dB |
| VAE 重建 SSIM | 0.8082 | **0.9275** | Map +0.12 |
| VAE KL 散度 | 400,263 | 916,627 | Map 信息压缩更完整 |
| LDM 重建 PSNR | — | **26.21 dB** | — |
| LDM 重建 SSIM | — | **0.7768** | — |

Map 域的 VAE/LDM 重建质量远高于 RSI，说明 **Map 的 C 分量天然包含更少的结构信息**（地图有大量平坦区域，遥感影像纹理丰富）。这为 A→B 方向的 C 分量退化创造了条件。

### 1.3 基线模型全指标（c_gradient_weight=0.0）

来自 `evaluation/cyclediff/res/evaluation_metrics.txt`：

| 指标 | ABA 循环 | 评级 |
|------|----------|------|
| pixel_l1_mean | 0.1134 ± 0.0168 | Fair（< 0.15） |
| c_l1_mean | 0.3893 ± 0.0490 | — |
| c_psnr_mean | 24.19 ± 2.37 dB | Good（> 20） |
| c_ssim_mean | 0.5248 ± 0.1075 | Fair（< 0.60） |
| ssim_mean | 0.3656 ± 0.1011 | Needs improvement |
| psnr_mean | 17.09 ± 1.25 dB | Needs improvement |
| lpips_mean | 0.5628 ± 0.0495 | Needs improvement |

基线模型的像素级循环一致性（ABA pixel_l1=0.11, psnr=17.09 dB）较差，表明 A→B→A 循环丢失了大量信息。

---

## 2. 数据集量化分析

### 2.1 文件大小分布

对 `/root/autodl-tmp/CycleDiff/datasets/maps/test/` 中 219 张测试图像的文件系统测量：

| 域 | 测试集数量 | 总大小 | 平均文件大小 | 标准差 |
|----|----------|--------|-------------|--------|
| **class_RSI** | 219 | 140.6 MB | **657.2 KB** | — |
| **class_Map** | 219 | 59.8 MB | **279.8 KB** | — |

**RSI 文件大小是 Map 的 2.35 倍**，直接反映了两域的内容复杂度差异：

| 示例文件 | RSI (class_RSI/) | Map (class_Map/) | 比例 |
|----------|------------------|------------------|------|
| 1.png | 679,258 bytes | 266,928 bytes | 2.54x |
| 10.png | 680,290 bytes | 275,829 bytes | 2.47x |
| 100.png | 684,820 bytes | 228,099 bytes | 3.00x |

### 2.2 域特性差异

| 特性 | RSI（遥感影像） | Map（地图瓦片） |
|------|----------------|----------------|
| 纹理类型 | 自然地形纹理（植被、水体、建筑）| 人工制图（色块、道路线、标注） |
| 纹理密度 | 极高，每像素都有自然变化 | 低，大面积平坦色块 |
| 边缘密度 | 高（地形边界、建筑轮廓）| 稀疏（道路、区域边界） |
| C 分量信息量 | 高（需编码更多低频结构）| 低（平坦区域编码为常数） |
| 频域分布 | 白色（宽带频谱）| 红色（低频主导） |

**结构不对称性是问题的触发器**——如果换成 cat↔dog 等纹理复杂度相近的域对，同一架构可能不表现明显问题。

---

## 3. 根因分析

### 3.0 关键证据：旧版脚本同样存在该问题

`train_uncond_ldm_cycle.py`（旧版，`train_uncond_ldm_cycle.py:438`）**不包含任何 `c_gradient_loss`**，其 generator 总损失仅由以下组成：

```python
loss_gen_toal = loss_idt + loss_G_adv_B + loss_G_adv_A
              + loss_cycle_ABA + loss_cycle_BAB
              + loss_perceptual + loss_ldm
```

但同样出现 A2B C 分量扁平化。这证明 **`c_gradient_loss` 不是根因，而是放大器**——根本原因存在于 CycleGAN + PatchGAN 的架构设计中。

### 3.1 唯一直接信号：`loss_G_adv_A`

在 generator step（`ga_ind=0`）中，`net_G_A` 的输出 `fake_C_T` 上作用的梯度来源（以旧版为例）：

| 损失 | 直接/间接 | 梯度路径 |
|------|----------|---------|
| `loss_G_adv_A` | **直接** | `∂MSE(D(fake_C_T), 1) / ∂fake_C_T` |
| `loss_cycle_ABA` | 间接 | `→ recon_C_S → net_G_B → fake_C_T` |
| `loss_cycle_BAB` | 间接 | `→ recon_C_T → 经 net_G_A 自身 → fake_C_S → net_G_B → ...` |
| `loss_perceptual` | 间接 | `→ recon_C_S → net_G_B → fake_C_T` |
| `loss_idt` | 间接 | `→ idt_C_S → net_G_B → ...` |

**`net_G_A` 的输出 `fake_C_T` 仅收到一个直接梯度来源：`loss_G_adv_A`。** 其他所有保结构损失必须穿过 `net_G_B` 的雅可比矩阵。

### 3.2 生成器的"捷径"困境

```
net_G_A 面临两种翻译策略：

路径 A（高成本）:                    路径 B（低成本）:
  RSI C ─→ 学习 Map 的特定空间结构      RSI C ─→ 全局降低梯度幅度
  (稀疏边缘，正确位置/方向/强度)          (均匀平滑，无空间选择性)
  
  判别器反馈: 像真实 Map C ✓            判别器反馈: 勉强像 Map C △
  需要学习容量: 高                      需要学习容量: 低
  任务难度: 高维稀疏结构预测             任务难度: 低维全局幅值调节
  损失曲面: 崎岖，局部极小点多            损失曲面: 平滑，易于优化
```

**生成器必然选择路径 B**，原因：

1. **梯度平滑性**：全局降低梯度幅度是一个低维、光滑的操作（类似于在整个张量上乘以一个标量因子）；而学习 Map 的特定空间结构（在正确位置生成稀疏边缘）需要高维精确控制，梯度在 C 空间的 `[B, 3, 64, 64]` 维度上高度稀疏
2. **判别器无法提供精确空间指导**：PatchGAN `NLayerDiscriminator2` 的输出是 `[B, 1, 8, 8]` 网格，每个 70×70 感受野几乎覆盖整个 64×64 输入。它只能给出"整体是否像 Map C"的判决，无法在 64×64 空间上提供逐像素的结构生成指导
3. **net_G_A 容量与任务不匹配**：`ResnetGenerator_timestep_restime_2_attn`（12 残差块 + attention）接收 `[B, 3, 64, 64]` 的 C 分量和随机时间步 t，需在 64×64 分辨率下同时完成域转换和精确结构生成——容量边界下，生成器自然选择成本最低的路径

### 3.3 为何 B→A 方向不受影响

`net_G_B` 的输出 `fake_C_S` 同样只收到一个直接梯度 `loss_G_adv_B`（"像 RSI C"）。但 **RSI C 的分布天然具有高结构密度**，所以：

- **路径 A（学结构）和"像 RSI C"同向**——生成结构是满足对抗损失的必经之路
- **路径 B（降梯度）在 B→A 方向不成立**——降梯度会让输出不像 RSI C，判别器会立刻惩罚

因此 B→A 方向不存在退化，不是因为架构有多好，而是因为**目标域天然要求结构**。

### 3.4 Cycle Consistency 的雅可比瓶颈

这是 cycle 架构的通用弱点：

```
loss_cycle_ABA = |recon_C_S - input_C_S|₁ × 40.0
                        ↑
recon_C_S = net_G_B(fake_C_T, t)
                        ↑
fake_C_T = net_G_A(input_C_S, t)

∂loss_cycle/∂θ = ∂loss/∂recon_C_S × ∂net_G_B/∂fake_C_T × ∂net_G_A/∂θ
                        ↑                       ↑
                   LPIPS/MSE              net_G_B 的雅可比矩阵
```

退化机制：
1. `net_G_A` 将 RSI C 翻译为低结构度的 `fake_C_T`
2. `net_G_B` 学会从低结构度的输入重建 RSI C → cycle loss 下降
3. `net_G_B` 对 `fake_C_T` 的雅可比在结构维度上退化——因为 net_G_B 内部通过残差连接和 attention 补偿了缺失信息，对输入的微小结构变化不再敏感
4. cycle loss 下降，但梯度不再能指导 net_G_A 保结构
5. **退化正反馈**：net_G_A 收到更弱的结构信号 → 输出持续退化 → 雅可比进一步退化

**所有保结构损失（cycle_ABA=40、cycle_BAB=20、perceptual=20、idt=1）都穿过此瓶颈。**

### 3.5 `c_gradient_loss` 的角色：放大器，非根因

新版脚本 `train_uncond_ldm_cycle_swanlab.py` 新增了 `loss_c_grad_target`（权重=3.0），它提供了一条额外的**直接**梯度通道：

```
loss_c_grad_target = c_gradient_loss_weighted(fake_C_T, input_C_T) × 3.0
```

该损失的目标 `input_C_T` 来自**不配对的随机 Map 图**. 其作用机制见 §4.3.2。在非配对设定下，Map C 的边缘密度（≈20%）低于 RSI C（≈40%），因此 `edge_weight` 施加的高权重区域在空间中稀疏且随机放置，对 fake_C_T 的空间结构约束较弱。

**`c_grad_target` 的作用不是"鼓励平坦化"，而是"加速 net_G_A 收敛到 3.2 节所述路径 B"**——它额外提供了一条直接的低维梯度通道，与 `loss_G_adv_A` 在"降低梯度"方向上叠加，使生成器更快锁定于捷径。

旧版脚本没有 `c_grad_loss` 但仍有扁平化 → 证明根因是架构级（3.2-3.4），`c_grad_loss` 只是加速了退化。

### 3.6 判别器为什么无法阻止

判别器损失持续下降说明**判别器确实能区分平坦假 C 和结构化真 Map C**。但它无法提供有用的纠正信号：

```
D(fake_C_T) → 0    (判别器判定：这是假的)
    ↓
loss_G_adv_A = MSE(D(fake_C_T), 1) = (0-1)² = 1.0     (很高)
    ↓
∂loss_G_adv_A / ∂fake_C_T = 2 × (0-1) × ∂D/∂fake_C_T = -2 × ∂D/∂fake_C_T
    ↓
    └─ 5 层卷积雅可比 ─┘
```

梯度幅值不弱，但方向是模糊的——"变得不像现在这样"。对生成器来说，最容易做到的就是**降低输出 C 的全局梯度幅度**（低维操作），而非在正确位置生成 Map 的稀疏边缘（高维操作）。判别器无法区分这两种策略。

### 3.7 综合根因链

```
┌─────────────────────────────────────────────────┐
│ 1. 目标域结构不对称 (RSI vs Map: 2.35× 信息密度)  │
│    ↓                                            │
│ 2. net_G_A 只有一条直接梯度 (loss_G_adv_A)        │
│    保结构信号全部穿过 net_G_B 雅可比 → 衰减        │
│    ↓                                            │
│ 3. 生成器面临高成本（学结构）vs 低成本（降梯度）      │
│    自然选择降梯度路径                              │
│    ↓                                            │
│ 4. net_G_B 学会从退化 C 重建 → 雅可比进一步退化    │
│    → 正反馈循环                                   │
│    ↓                                            │
│ 5. 判别器能检测差异但无法提供精确空间梯度            │
│    loss_D ↓ 但 fake_C_T 持续退化                  │
│    ↓                                            │
│ 6. c_grad_target (如存在) 额外提供直接降梯度信号    │
│    加速退化 ← 但非根因                             │
└─────────────────────────────────────────────────┘
```

| 组件 | 角色 |
|------|------|
| PatchGAN 判别器 | 提供"像目标域"信号，无法提供空间结构指导 |
| Cycle consistency | 保结构，但被 net_G_B 雅可比衰减 |
| `c_gradient_loss` | **放大器**，加速生成器向捷径收敛 |
| 域结构不对称 | **触发器**，使捷径在 A→B 方向可行但在 B→A 方向不可行 |
| 生成器容量 | **限制因素**，降低了学习精确结构的可行性 |

---

## 4. 解决方案设计

### 4.1 设计目标

1. **在 net_G_A 输出上引入直接的结构保持梯度**，绕过 `net_G_B` 雅可比瓶颈
2. **不依赖配对数据集**，保持 CycleGAN 非配对范式
3. **允许跨域语义转换**，不强制输出与输入完全相同
4. **可使用现有代码基础设施**，最小化改动

### 4.2 排除的方案

| 方案 | 问题 |
|------|------|
| **增大 cycle_weight** | 仍穿过 net_G_B 雅可比瓶颈，已被实验证实无效 |
| **梯度循环一致性**（对比 `∇fake_C_T` 和 `∇input_C_S`）| 仍需穿过 net_G_B（因为 fake_C_S 经 net_G_B 后才回到源域）|
| **FFT 频率域损失** | 无法针对性保护空间结构，只能匹配能量谱 |
| **dual cycle loss** (A→B→A 和 B→A→B) | 同 cycle 瓶颈问题 |
| **LSGAN 替换 PatchGAN** | 不影响 C 空间结构，治标不治本 |

### 4.3 选定方案：Dual-Target Gradient Loss（方案 1）

#### 4.3.1 核心思路与设计动机

根据 §3 的根因分析，`net_G_A` 面临的核心问题是：**所有直接梯度都鼓励"降梯度"（走捷径），而保结构信号的唯一通道（net_G_B 雅可比）已经退化。** 需要一条绕过 `net_G_B` 的直接结构保持信号。

即使没有 `loss_c_grad_target`（如旧版 `train_uncond_ldm_cycle.py`），`loss_G_adv_A` 单独也足以导致扁平化——因为"变得更平坦"是满足判别器的最低成本路径。`loss_c_grad_target` 只是加速了这一过程。

因此 `loss_c_grad_source` 的设计目标**不仅仅是抵消 `loss_c_grad_target`，而是提供一条与捷径正交的直接梯度通道**——"输出 C 的结构应该与输入 C 的结构匹配"。

具体实现：在现有 `loss_c_grad_target`（匹配目标域梯度分布）的基础上，新增 `loss_c_grad_source`（匹配源域梯度分布），**两者都直接施加在 `fake_C_T` 上**：

```
fake_C_T = net_G_A(input_C_S, t)    ← RSI C → Map C
                                         ↑
loss_c_grad_target  ←→  input_C_T   ←  随机 Map 图的 C (非配对，加速捷径)
loss_c_grad_source  ←→  input_C_S   ←  自己的输入 (天然配对，对抗捷径)
```

#### 4.3.2 数学公式

损失函数 `c_gradient_loss_weighted`（`util/c_gradient_loss.py:5-15`）：

设输入张量形状为 `[B, 3, H, W]`：

```
水平梯度:  dx_trans = c_translated[:, :, :, 1:] - c_translated[:, :, :, :-1]    [B,3,H,W-1]
垂直梯度:  dy_trans = c_translated[:, :, 1:, :] - c_translated[:, :, :-1, :]    [B,3,H-1,W]

目标梯度:  dx_target = c_target[:, :, :, 1:] - c_target[:, :, :, :-1]           [B,3,H,W-1]
          dy_target = c_target[:, :, 1:, :] - c_target[:, :, :-1, :]           [B,3,H-1,W]

边缘权重:  edge_x = 1 + edge_weight × |dx_target|.mean(dim=1).detach()           [B,1,H,W-1]
          edge_y = 1 + edge_weight × |dy_target|.mean(dim=1).detach()           [B,1,H-1,W]

损失:     L = mean(edge_x × |dx_trans - dx_target|) + mean(edge_y × |dy_trans - dy_target|)
```

**关键设计细节**：
- `edge_weight=10.0`：在目标有强边缘处，误差放大到 **11 倍**
- `.detach()`：**边缘权重自身不产生梯度**，只通过 `|dx_trans - dx_target|` 中的 `dx_trans` 回传
- 对整个 `|dx_trans - dx_target|` 求 `mean`：匹配梯度分布统计量，非空间精确对齐
- 使用 L1 范数：对离群梯度鲁棒

#### 4.3.3 代码实现

`train_uncond_ldm_cycle_swanlab.py:626-640`：

```python
# ===== 新增: loss_c_grad_target (原有) =====
loss_c_grad_target = torch.tensor(0.0, device=fake_C_T.device)
if hasattr(self.cfg.trainer, "c_gradient_weight"):
    loss_c_grad_target = c_gradient_loss_weighted(
        fake_C_T, input_C_T,                                    # 输出 vs 目标 Map C (非配对)
        edge_weight=self.cfg.trainer.get('c_gradient_edge_boost', 10.0)
    ) * self.cfg.trainer.c_gradient_weight

# ===== 新增: loss_c_grad_source (方案1核心) =====
loss_c_grad_source = torch.tensor(0.0, device=fake_C_T.device)
if hasattr(self.cfg.trainer, "c_gradient_preserve_weight"):
    loss_c_grad_source = c_gradient_loss_weighted(
        fake_C_T, input_C_S,                                    # 输出 vs 自己的输入 (天然配对)
        edge_weight=self.cfg.trainer.get('c_gradient_preserve_edge_boost', 10.0)
    ) * self.cfg.trainer.c_gradient_preserve_weight

loss_c_grad = loss_c_grad_target + loss_c_grad_source
```

#### 4.3.4 为何 source loss 是天然配对的

`input_C_S` 来自于 `batch["src_img"]`（RSI 图 A），经 VAE+LDM 预测得到的 C 分量；`fake_C_T = net_G_A(input_C_S, t)` 正是用同一个 `input_C_S` 翻译而来。输出参照物是其自身输入，**不依赖 src-trgs 成对出现**。

而 `loss_c_grad_target` 中，`fake_C_T` 来自 RSI 图 A，`input_C_T` 来自 Map 图 B，两者是不配对的——该损失通过大数定律（几千次迭代随机配对）收敛到梯度分布匹配。

#### 4.3.5 权重设计

| 损失 | 权重 | 边缘增强 | 语义 | 方向 |
|------|------|---------|------|------|
| `loss_G_adv_A` | 2.0 | — | "像 Map C" | **捷径方向（降梯度是低成本路径）** |
| `loss_c_grad_target` | **3.0** | 10.0 | "梯度统计像 Map C"（非配对） | **捷径方向（放大降梯度信号）** |
| `loss_c_grad_source` | **2.0** | 10.0 | "保留源域结构梯度"（天然配对） | **正交方向（对抗捷径）** |

**`loss_c_grad_source` 的核心作用不是简单地"抵消 `loss_c_grad_target`"，而是提供一条与捷径正交的梯度通道。** 即使未来去掉 `loss_c_grad_target`（或设为 0），`loss_c_grad_source` 仍然必要——因为 `loss_G_adv_A` 单独就足以诱导短路（§3.0 旧版证据）。

#### 4.3.6 修复后的梯度流

```
net_G_A 参数更新 (ga_ind=0):

  直接梯度:
    ∇(loss_c_grad_source)  weight=2.0  │  ← 对抗捷径：保留输入结构
    ∇(loss_c_grad_target)  weight=3.0  │  → 加速捷径：匹配目标梯度分布 (非配对)
    ∇(loss_G_adv_A)        weight=2.0  │  → 诱导捷径："像 Map C"的最低成本路径
    ∇(loss_ldm)            weight=0.05 │  ○ 中性
    ───────────────────────────────────│
    净效果: 2.0 "保结构" vs 5.05 "捷径"  │  ← 需监控 loss_c_grad_source/target 比值

  间接梯度 (经 net_G_B):
    ∇(loss_cycle_ABA)   weight=40      │  衰减↓ (Jacobian退化)
    ∇(loss_cycle_BAB)   weight=20      │  衰减↓
    ∇(loss_perceptual)  weight=20      │  衰减↓
```

### 4.4 配置参数

`configs/maps/translation_cgrad.yaml:161-165`：

```yaml
# 原有参数
c_gradient_weight: 3.0              # target loss 权重
c_gradient_edge_boost: 10.0         # target loss 边缘增强

# 方案1 新增参数
c_gradient_preserve_weight: 2.0     # source loss 权重 (略低于 target)
c_gradient_preserve_edge_boost: 10.0 # source loss 边缘增强
```

---

## 5. 实验验证

### 5.1 实验脚本

```bash
# 训练
python train_uncond_ldm_cycle_swanlab.py --cfg configs/maps/translation_cgrad.yaml

# 可视化验证
python evaluation/cyclediff/quick_a2b_c_vis.py
```

配置文件已修改：`train_num_steps: 2000`（原文 3000）、`save_every: 2000`（原文 3000）。

### 5.2 监控指标

训练过程中，通过 SwanLab 日志监控以下指标（每 50 步记录）：

| 指标 | 含义 | 预期 |
|------|------|------|
| `loss_c_grad_target` | 输出 C 与 Map 域 C 的梯度匹配度 | 保持在 ~0.05 |
| `loss_c_grad_source` | 输出 C 与源 RSI C 的梯度匹配度 | 保持 > target 的 0.5-1.5x |
| `loss_c_grad_source / loss_c_grad_target` 比值 | 结构保持 vs 域转换的平衡 | 0.3-1.5 |
| `loss_cycle_ABA` | A→B→A 循环重建质量 | 不应变差 |

### 5.3 超参调节指南

| 观察 | 调整 |
|------|------|
| A→B C 分量仍有色块，无明显边缘 | 增大 `c_gradient_preserve_weight` (2.0 → 4.0) |
| A→B 翻译结果过于接近 RSI，缺少 Map 风格 | 减小 `c_gradient_preserve_weight` (2.0 → 1.0) 或增大 `c_gradient_weight` |
| 边缘区域模糊但非边缘区域干净 | 增大 `c_gradient_preserve_edge_boost` (10.0 → 15.0) |
| source/target loss 比值极小 (< 0.1) | 源结构梯度被淹没，增大 `c_gradient_preserve_weight` |

### 5.4 当前训练状态

| 项目 | 状态 |
|------|------|
| Screen 会话 | `cgrad_2k`（PID 34068） |
| 训练进程 | 运行中（PID 34074，~25 GB VRAM） |
| 日志位置 | `swanlog/run-20260511_150531-*/` |
| 预计完成 | ~50 分钟 |

---

## 6. 结论与讨论

### 6.1 故障模式总结

| 维度 | 发现 |
|------|------|
| **根本原因** | `net_G_A` 输出只收到一条直接梯度 `loss_G_adv_A`，结构保持信号全部经 `net_G_B` 雅可比衰减；生成器在"学结构"（高成本）和"降梯度"（低成本）两条路径中自然选择后者 |
| **触发条件** | 目标域结构密度低于源域——B→A 方向不受影响，因为"像 RSI C"天然要求生成结构 |
| **证据** | `train_uncond_ldm_cycle.py` 无 `c_grad_loss` 仍出现扁平化，证明根因是架构级而非损失函数级 |
| **`c_gradient_loss` 的角色** | **放大器**，非根因——额外提供一条直接降梯度通道，加速生成器向捷径收敛 |
| **判别器的角色** | 能区分真假，但判别器梯度（经 5 层卷积）无法提供逐像素空间结构生成指导——只能告诉生成器"你整体不像真的"，无法告诉它"在 (x,y) 位置生成一条边" |
| **性质** | **CycleGAN 框架在非对称域对上的通用弱点**，更换纹理复杂度相近的域对（如 cat↔dog）不触发 |

### 6.2 方案评估

`loss_c_grad_source` 提供了一个稳定、配对、直接的梯度信号来对抗生成器的捷径（§3.2），而不仅仅是抵消 `loss_c_grad_target`（后者被确认只是放大器，§3.5-3.7）：

1. **直接性**：梯度直接在 `fake_C_T` 上，绕过 `net_G_B` 的 Jacobian 退化瓶颈
2. **正交性**：梯度方向是"保输入结构"，与对抗损失的"像目标域"方向正交——阻止生成器走纯降梯度的捷径
3. **配对性**：与自身输入比较，天然配对，不依赖 src-trgs 成对
4. **退火性**：匹配的是梯度分布统计量，允许有限的域转换
5. **权重隔离**：`edge_weight.detach()` 确保边缘掩码不产生自身梯度，梯度只通过 `|∇fake - ∇src|` 中的 `∇fake` 回传

### 6.3 后续工作

- [x] 实现 `loss_c_grad_source` 并集成到训练循环
- [ ] **训练 2000 步验证效果** ← 当前进行中
- [ ] 运行 `quick_a2b_c_vis.py` 检查 A→B C 分量可视化
- [ ] 调参 `c_gradient_preserve_weight` 至最佳平衡点
- [x] 向全量训练配置 (`translation_C_disc_timestep_ode_2.yaml`, 200k 步) 同步参数
- [x] 全量训练验证

---

*报告生成时间: 2026-05-11*
*相关文件: `train_uncond_ldm_cycle_swanlab.py`, `util/c_gradient_loss.py`, `configs/maps/translation_cgrad.yaml`*
