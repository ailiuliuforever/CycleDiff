# 非配对 Feature Matching 实现方案

> **状态**: 设计文档，未实施
> **目标**: 在判别器中间特征空间添加分布匹配约束，对抗生成器走"降梯度"捷径

---

## 1. 动机

### 1.1 问题回顾

根据 `analysis_report.md` §3.2，`net_G_A` 在 A→B 翻译中选择**低成本捷径**（全局降低 C 分量梯度），导致结构退化。当前 SSIM 在 cycle 重建端提供约束（已验证有效），但仍有不足：

| 现有约束 | 位置 | 针对的退化维度 | 弱点 |
|---------|------|-------------|------|
| `loss_c_ssim` (cycle) | C 空间 | 空间结构 | 经 net_G_B 雅可比 |
| `loss_G_adv_A` | D 最终标量输出 | 域分布 | 1 个标量/感受野，信息带宽极低 |
| `loss_perceptual` (LPIPS) | 经 net_G_B | 感知质量 | 雅可比衰减 |

Feature Matching 在 D 的**中间特征空间**提供多层纹理分布约束，填补 SSIM 和对抗损失之间的空白。

### 1.2 设计原则

| 原则 | 实现 |
|------|------|
| **非配对友好** | `L1(feats_fake, feats_real.detach())` + `mean()` 压扁空间维度 |
| **最小侵入** | 仅修改 D 的 `forward` 返回值，不改变现有损失计算 |
| **可配置** | 新参数集中管理，默认关闭（weight=0） |
| **双向对称** | 同时对 D_A 和 D_B 做 FM |
| **信息带宽最大化** | 收集 4 层中间特征（64ch→128ch→256ch→512ch） |

### 1.3 预期效果

- **对抗降梯度短路**: D 的第一层卷积学到边缘/纹理检测器，平坦 C 在该层响应接近零 → 巨大 L1
- **增强纹理分布匹配**: 4 层特征提供 ~147K 个约束元素 vs 最终标量输出的 64 个
- **稳定 G 训练**: 多信号叠加减少模式漂移

---

## 2. 架构分析

### 2.1 判别器层结构

`NLayerDiscriminator2` (`ddm/cycle_discriminator.py:44-101`)，`n_layers=3, no_antialias=False`：

| 索引 | 模块 | 输出形状 | 特征语义 |
|------|------|---------|---------|
| 0-2 | Conv(3→64), LReLU, Downsample | `[B, 64, 32, 32]` | 低级纹理（边缘方向/频率） |
| 3-6 | Conv(64→128), InstNorm, LReLU, Downsample | `[B, 128, 16, 16]` | 中级纹理（局部模式/角点） |
| 7-10 | Conv(128→256), InstNorm, LReLU, Downsample | `[B, 256, 8, 8]` | 高级结构（区域形状） |
| 11-13 | Conv(256→512), InstNorm, LReLU | `[B, 512, 8, 8]` | 最抽象特征（全局上下文） |
| 14 | Conv(512→1) | `[B, 1, 8, 8]` | 最终标量输出 |

`self.model` 是 `nn.Sequential`，15 个模块，无命名子层。

### 2.2 特征收集策略

```python
def forward(self, input, return_features=False):
    if not return_features:
        return self.model(input)  # 原始行为，零开销
    features = []
    x = input
    # 在第 2, 6, 10, 13 个模块之后收集特征
    # (对应 LReLU 激活之后、下一层 Downsample 之前)
    for i, layer in enumerate(self.model):
        x = layer(x)
        if i in [2, 6, 10, 13]:
            features.append(x)
    return x, features
```

**设计决策**：收集 LeakyReLU **之后**的特征（已非线性激活，表达能力更强），而非之前的线性特征。

### 2.3 FM 损失函数

```python
def discriminator_feature_matching_loss(feats_fake, feats_real):
    """
    Unpaired feature matching in discriminator activation space.
    
    Args:
        feats_fake: list of [B, C_i, H_i, W_i] tensors from fake input
        feats_real: list of [B, C_i, H_i, W_i] tensors from real input
    Returns:
        scalar loss
    """
    loss = 0.0
    for f_fake, f_real in zip(feats_fake, feats_real):
        # mean() over all spatial+channel+spatial dims
        # → distribution matching, not spatial alignment
        loss += F.l1_loss(f_fake, f_real.detach())
    return loss
```

**关键**：`L1_loss` 默认 `reduction='mean'`，对所有 `B×C×H×W` 元素求均值——**天然适配非配对场景**，不要求空间对应，只要求特征激活的统计分布一致。

---

## 3. 实现计划

### 文件 1: `ddm/cycle_discriminator.py`

**修改点 1: `forward` 方法 (line 99-101)**

```python
# === 修改前 ===
def forward(self, input):
    """Standard forward."""
    return self.model(input)

# === 修改后 ===
def forward(self, input, return_features=False):
    """Forward pass with optional feature extraction."""
    if not return_features:
        return self.model(input)
    features = []
    x = input
    for i, layer in enumerate(self.model):
        x = layer(x)
        # Collect after activation at: Conv+ReLU+Down, Conv+Norm+ReLU+Down, ...
        if i in [2, 6, 10, 13]:
            features.append(x)
    return x, features
```

**影响评估**:
- `return_features=False`（默认）: 行为与修改前**完全一致**，零性能开销
- `return_features=True`: 仅在新 loss 计算时才启用
- 特征收集用 `for i, layer in enumerate(self.model)`: 避免硬编码子模块名称，`nn.Sequential` 的直接索引稳定

**修改点 2: `__init__` — 不需要改变**

---

### 文件 2: `train_uncond_ldm_cycle_swanlab.py`

共 8 处修改，按执行顺序：

#### 修改 2.1: 导入 FM 损失函数 (line 18 之后)

```python
# 新增导入
from util.c_fm_loss import discriminator_feature_matching_loss
```

#### 修改 2.2: 计算 FM 损失 — ga_ind=0 (line 631 之后, 633 之前)

```python
# === 在 loss_G_adv_B (line 631) 之后、loss_perceptual (line 633) 之前插入 ===

                # Feature Matching: enforce D intermediate activation match
                loss_G_fm = torch.tensor(0.0, device=fake_C_T.device)
                if self.cfg.trainer.get('fm_weight', 0.0) > 0:
                    _, feats_fake_A = self.net_D_A(fake_C_T, return_features=True)
                    _, feats_real_A = self.net_D_A(input_C_T, return_features=True)
                    loss_G_fm_A = discriminator_feature_matching_loss(
                        feats_fake_A, feats_real_A
                    )
                    _, feats_fake_B = self.net_D_B(fake_C_S, return_features=True)
                    _, feats_real_B = self.net_D_B(input_C_S, return_features=True)
                    loss_G_fm_B = discriminator_feature_matching_loss(
                        feats_fake_B, feats_real_B
                    )
                    loss_G_fm = (loss_G_fm_A + loss_G_fm_B) * self.cfg.trainer.fm_weight
```

**关键设计**:
- `input_C_T` 和 `input_C_S` 都已经被 `self.net_D_A` / `self.net_D_B` 在 ga_ind=1（判别器阶段）使用过，D 已学会其特征分布
- FM 在 ga_ind=0（生成器阶段）计算，**梯度流向生成器**（`fake_C_T` → `net_G_A`）
- `.detach()` 在 `L1_loss` 内部对 `f_real` 使用，确保梯度只通过 `f_fake` 回传

#### 修改 2.3: 加入总损失 (line 649)

```python
# === 修改前 ===
loss_gen_toal = loss_idt + loss_G_adv_B + loss_G_adv_A + loss_cycle_ABA + loss_cycle_BAB + loss_perceptual + loss_ldm + loss_c_ssim

# === 修改后 ===
loss_gen_toal = loss_idt + loss_G_adv_B + loss_G_adv_A + loss_cycle_ABA + loss_cycle_BAB + loss_perceptual + loss_ldm + loss_c_ssim + loss_G_fm
```

#### 修改 2.4: 加入 loss_dict (line 657 之后, 658 之前)

```python
                   "{}/loss_c_ssim".format(split): loss_c_ssim.detach(),
                   "{}/loss_G_fm".format(split): loss_G_fm.detach(),     # 新增
                   "{}/loss_ldm".format(split): loss_ldm.detach(),
```

#### 修改 2.5: pbar 日志解包 (line 794 之后, 795 之前)

```python
                             loss_c_ssim = log_dict["train/loss_c_ssim"]
                             loss_G_fm = log_dict["train/loss_G_fm"]        # 新增
                             loss_gen_toal = log_dict["train/loss_gen_toal"]
```

#### 修改 2.6: TensorBoard 日志 (line 866 之后, 867 之前)

```python
                     self.writer.add_scalar('Generator/loss_c_ssim', loss_c_ssim, self.step)
                     self.writer.add_scalar('Generator/loss_G_fm', loss_G_fm, self.step)    # 新增
                     self.writer.add_scalar('Generator/loss_gen_toal', loss_gen_toal, self.step)
```

#### 修改 2.7: SwanLab 日志 (line 884 之后, 885 之前)

```python
                             "Generator/loss_c_ssim": loss_c_ssim.item(),
                             "Generator/loss_G_fm": loss_G_fm.item(),         # 新增
                         "Discriminator/loss_dis_total": loss_dis_total.item(),
```

---

### 文件 3: `util/c_fm_loss.py` (新建)

```python
import torch.nn.functional as F


def discriminator_feature_matching_loss(feats_fake, feats_real):
    """
    Unpaired feature matching in discriminator activation space.

    Computes L1 distance between fake and real intermediate features
    of the discriminator. The mean() over all spatial+channel+batch dims
    makes this a distribution-matching loss, compatible with unpaired data.

    Args:
        feats_fake: list of [B, C_i, H_i, W_i] feature tensors from fake input
        feats_real: list of [B, C_i, H_i, W_i] feature tensors from real input
    Returns:
        scalar loss = mean(|fake_feat - real_feat.detach()|)
    """
    loss = 0.0
    for f_fake, f_real in zip(feats_fake, feats_real):
        loss += F.l1_loss(f_fake, f_real.detach())
    return loss
```

---

### 文件 4: `configs/maps/translation_cgrad.yaml`

在 line 185（`d_B_weight: 1.5`）之后添加：

```yaml
  # for discriminator feature matching loss (generator side)
  fm_weight: 2.0
```

**推荐初始 weight=2.0**:
- 与 `g_adv_A_weight: 2.0` 同级，避免过渡约束
- FM 损失天然比对抗损失大（4 层 L1 vs 1 个标量 MSE），无需高权重
- 设为 0 即完全禁用

---

## 4. 在非配对场景如何工作

### 4.1 与其他非配对损失的对比

| 损失 | 空间信息 | 配对要求 | 信息带宽 | 抗捷径能力 |
|------|---------|---------|---------|---------|
| LSGAN 对抗（输出标量） | 无（1 标量/patch） | 无 | 极低（64 元素） | 弱 |
| MCL（对比学习） | 无（flatten） | 无 | 低 | 弱（weight=0.01） |
| `c_gradient_loss` | 有（梯度 L1） | 无（非配对，mean） | 中 | 被分布匹配作弊 |
| **FM（中间特征）** | **部分保留（空间网格）** | **无（mean）** | **极高（~147K）** | **强** |
| SSIM（cycle 重建） | 强（5×5 窗口） | 有（循环路径配对） | 高 | 最强 |

### 4.2 信息带宽分析

```
LSGAN:  D(fake) → [B,1,8,8] → 64 个约束元素
FM:     D.layer0 → [B,64,32,32]  → 65,536
        D.layer1 → [B,128,16,16] → 32,768
        D.layer2 → [B,256,8,8]   → 16,384
        D.layer3 → [B,512,8,8]   → 32,768
        ─────────────────────────────────
        合计:                       ~147,456 个约束元素

带宽比 = 147,456 / 64 = 2,304 倍
```

**高带宽让生成器无法通过简单降方差同时满足所有层的约束。**

### 4.3 为什么分布匹配可行

```
fake_C_T 来自 RSI 图 A（翻译后）
real_C_T 来自 Map 图 B（随机另一张 Map）

D.layer0 的每个卷积核是一个纹理检测器（学自 Map C 分布）
  → "水平边缘检测器"在真 Map C 某处激活 ~0.8
  → 同一检测器在假 C 处激活 ~0.1（因为边缘被抹平）
  → L1 = |0.1 - 0.8| = 0.7 → 巨大梯度迫使生成器保留边缘

mean() 压扁空间 → "不管边缘在哪里，只要有的地方有边缘就行"
             → 适配非配对场景
             → 仍比"不管有没有边缘"的最终标量强 2304 倍
```

---

## 5. 计算开销

| 开销来源 | 额外时间 | 额外显存 |
|---------|---------|---------|
| D 前向 ×2（真假各一次）| ~5ms/step | ~200MB（D 中间特征） |
| L1 loss ×4 层 | ~1ms/step | 可忽略 |
| **总计** | **~6ms/step** | **~200MB** |

当前每步 ~120ms（含 G+D+LDM 更新），FM 增加 ~5%。

---

## 6. 监控指标

| 指标 | 含义 | 预期 |
|------|------|------|
| `loss_G_fm` | FM 损失（4 层 L1 平均） | 随训练从 ~5 降到 ~1-2 |
| `loss_G_fm / loss_G_adv_A` | FM 与对抗损失的比例 | 2-3x（正常） |
| `loss_c_ssim` | 应继续下降（FM 与 SSIM 互补） | 比无 FM 时更低 |
| A2B 梯度幅值 | 应在 FM + SSIM 叠加下 > 15.5 | 当前纯 SSIM 为 14.58 |

---

## 7. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| D 前向 ×2 增加梯度图复杂性 | 中 | `retain_graph=True` 已在使用，增加一条前向路径不影响 |
| FM 权重过高导致 overfit 到一批 Map C 特征 | 低 | weight=2.0，与对抗损失同级 |
| `.detach()` 位置错误导致 D 参数被更新 | 低 | `f_real.detach()` 明确在 loss 函数内 |
| 4 层特征量级不同导致高层主导 | 低 | 每层独立 L1（自动归一化），不做加权求和 |
| 与 SSIM 梯度冲突 | 极低 | SSIM 保空间结构，FM 保纹理分布——互补非竞争 |

---

## 8. 与已有方案的协同

```
net_G_A 梯度信号谱 (ga_ind=0, A2B 方向):

  空间结构保持 ─┬─ c_ssim (cycle, w=15)  ──→ 强制局部对齐
              ├─ cycle_ABA (L1, w=40) ──→ 经 Jacobian 衰减
              └─ idt (L1, w=1)        ──→ 弱

  纹理分布保持 ─┬─ FM (D 中间特征, w=2) ──→ 多层分布匹配 ← 新增
              └─ perceptual (LPIPS, w=20) ──→ 经 Jacobian 衰减

  域分布逼近 ─┬─── G_adv_A (D 输出, w=2)  ──→ 基础对抗
             └─── G_adv_B (D 输出, w=1)  ──→ 基础对抗

  捷径路径 ────── (无直接梯度，需被上述约束间接阻止)
```

**FM 填补了空间结构（SSIM）和域分布（对抗）之间的中间层 —— 纹理/模式的分布匹配。**

---

## 9. 实施检查清单

| 步骤 | 文件 | 修改内容 |
|------|------|---------|
| 1 | `util/c_fm_loss.py` | 新建 FM 损失函数 |
| 2 | `ddm/cycle_discriminator.py` | `forward()` 添加 `return_features` 参数 |
| 3 | `train_uncond_ldm_cycle_swanlab.py:19` | 导入 `discriminator_feature_matching_loss` |
| 4 | `train_uncond_ldm_cycle_swanlab.py:631后` | 计算 `loss_G_fm` |
| 5 | `train_uncond_ldm_cycle_swanlab.py:649` | 加入总损失 |
| 6 | `train_uncond_ldm_cycle_swanlab.py:657` | 加入 loss_dict |
| 7 | `train_uncond_ldm_cycle_swanlab.py:794` | pbar 日志解包 |
| 8 | `train_uncond_ldm_cycle_swanlab.py:866` | TensorBoard 日志 |
| 9 | `train_uncond_ldm_cycle_swanlab.py:884` | SwanLab 日志 |
| 10 | `configs/maps/translation_cgrad.yaml:185后` | 添加 `fm_weight: 2.0` |

---

## 10. 训练命令（实施后）

```bash
python train_uncond_ldm_cycle_swanlab.py --cfg configs/maps/translation_cgrad.yaml
```

已有配置将与 FM 损失叠加：
- `c_ssim_weight: 15.0`（SSIM）
- `fm_weight: 2.0`（新增 FM）
- `train_num_steps: 5000`

---
*方案生成时间: 2026-05-11*
