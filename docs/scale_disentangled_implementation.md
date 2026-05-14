# 尺度解耦翻译 — 最小可行实现计划

> **状态**: 计划文档，准备实施
> **基于**: 诊断实验 `verify_scale_hypothesis.py` 的结果（grad_fake ≈ 0.075 在所有 t 上恒定）
> **目标**: 用最小改动实现尺度解耦，快速验证核心假说

---

## 0. 诊断实验结论

| 发现 | 含义 |
|------|------|
| `grad_fake ≈ 0.075` 在所有 t 上恒定 | 生成器的FiLM形同虚设——t信息被忽略 |
| 精细输入（t=0.05, grad=1.18）被压缩15× | 精细结构完全丢失 |
| 粗粒输入（t=0.90, grad=0.09）几乎不变 | 已接近生成器的恒定输出水平 |

**根因**：统一损失函数对所有 t 一视同仁。最优策略 = 输出恒定梯度水平的C分量。

**尺度解耦的核心干预**：让损失函数依赖于t——精细t施加强SSIM，粗粒t施加强对抗。

---

## 1. 最小可行版本（2尺度设计）

### 为什么不从3尺度开始？

| | 3尺度（原设计） | 2尺度（MVBP） |
|---|---|---|
| 额外UNet前向 | +200% (x6) | **+100% (x4)** |
| 额外G前向 | +200% (x6) | **+100% (x4)** |
| 代码改动 | 50行训练循环 | **25行训练循环** |
| 验证周期 | 5小时训练 | **3小时训练** |
| 风险 | 中等（3个尺度可能冲突） | **低（精细vs粗粒耦合清晰）** |

**2尺度足以验证核心假说**：如果生成器能学会在精细t输出高梯度、在粗粒t输出低梯度，则3尺度扩展是trivial的。

### 尺度划分

```
scale_id=0 (精细):  t ∈ [eps, 0.5]  →  SSIM约束主导 → 输出应保留结构
scale_id=1 (粗粒):  t ∈ [0.5, 1.0]  →  对抗约束主导 → 输出应匹配目标域
```

---

## 2. 文件改动清单

### 文件1: `ddm/cycle_generator_2.py`

#### 改动1a: `__init__` 添加scale_id注入（~8行，line 393之后）

```python
# 在 self.time_mlp 定义之后添加 (line 398之后)

# Scale-ID embedding for scale-disentangled translation
self.scale_emb = nn.Embedding(2, 64)  # 2个尺度: fine=0, coarse=1
self.scale_proj = nn.Linear(time_dim + 64, time_dim)  # 576→512
```

#### 改动1b: `forward` 签名和time_emb注入（~6行，lines 436-454）

```python
def forward(self, input, time, scale_id=None):
    """Standard forward with optional scale-disentanglement
    
    Args:
        input: [B, 3, H, W] input C tensor
        time:  [B] diffusion timestep
        scale_id: [B] long tensor, optional scale id (0=fine, 1=coarse)
    """
    t = self.time_mlp(time)               # [B, 512]
    
    # Scale-disentangled conditioning
    if scale_id is not None:
        s = self.scale_emb(scale_id)      # [B, 64]
        t = torch.cat([t, s], dim=-1)     # [B, 576]
        t = self.scale_proj(t)            # [B, 512] 投影回原维度
    
    x = self.init_conv(input)
    # ... rest unchanged, uses t as before
```

**关键设计**：
- `scale_id=None`（默认）→ 行为与当前完全一致 → 向后兼容
- `scale_id`的embedding与`time_emb`在**输入MLP之前**拼接 → 共享FiLM通道 → 最小参数增量
- `scale_proj`是`Linear(576, 512)` → 仅增加 576×512 = 295K 参数（可忽略）

---

### 文件2: `train_uncond_ldm_cycle_swanlab.py`

#### 改动2a: 精细化timestep采样（~10行，替换 line 600）

```python
# 当前 (line 600):
t = torch.rand(x_s.shape[0], device=x_s.device) * (1. - eps) + eps

# 替换为:
t_raw = torch.rand(x_s.shape[0], device=x_s.device) * (1.0 - eps) + eps

# 2尺度设计: 50%概率采样精细t, 50%概率采样粗粒t
use_coarse = (torch.rand(1, device=x_s.device) > 0.5).item()
if use_coarse:
    t = 0.5 + t_raw * 0.5                          # [0.5, 1.0]
    scale_id_A = torch.ones(x_s.shape[0], dtype=torch.long, device=x_s.device)  # coarse=1
else:
    t = eps + t_raw * 0.5                          # [eps, 0.5]
    scale_id_A = torch.zeros(x_s.shape[0], dtype=torch.long, device=x_s.device) # fine=0
```

**设计说明**：每个batch随机选择精细或粗粒模式。生成器在每个batch只看到一种尺度——学到了精细模式就不会在每个batch被粗粒模式的梯度"拉扯"。两个行为通过不同的scale_id解耦。

#### 改动2b: 带scale_id的翻译（~5行，替换 lines 609-623）

```python
# 所有 net_G_A 调用添加 scale_id
fake_C_T = self.net_G_A(input_C_S, t, scale_id=scale_id_A)
recon_C_S = self.net_G_B(fake_C_T, t)              # net_G_B 不收 scale_id
idt_C_S = self.net_G_B(input_C_S, t)

# 目标方向: net_G_A 也带 scale_id
fake_C_S = self.net_G_B(input_C_T, t)              # net_G_B 不收 scale_id
recon_C_T = self.net_G_A(fake_C_S, t, scale_id=scale_id_A)
idt_C_T = self.net_G_A(input_C_T, t, scale_id=scale_id_A)
```

**设计决定**：`net_G_B` **不**接收 `scale_id`。原因：
1. B→A方向（Map→RSI）不存在结构退化问题
2. 减少改动量 → 最小化风险
3. net_G_B保持现有行为（已能够保结构）

#### 改动2c: 尺度相关损失权重（~15行，替换 lines 636-668）

```python
# 尺度相关权重
if use_coarse:
    # 粗粒模式: 域转换主导
    w_ssim    = self.cfg.trainer.get('scale_coarse_ssim_weight', 3.0)
    w_adv     = self.cfg.trainer.get('scale_coarse_adv_weight', 3.0)
    w_cycle   = self.cfg.trainer.get('scale_coarse_cycle_weight', 20)
    w_percep  = self.cfg.trainer.get('scale_coarse_percep_weight', 10)
else:
    # 精细模式: 结构保持主导
    w_ssim    = self.cfg.trainer.get('scale_fine_ssim_weight', 15.0)
    w_adv     = self.cfg.trainer.get('scale_fine_adv_weight', 0.0)
    w_cycle   = self.cfg.trainer.get('scale_fine_cycle_weight', 40)
    w_percep  = self.cfg.trainer.get('scale_fine_percep_weight', 20)

# 带权重的损失
loss_cycle_ABA = F.l1_loss(recon_C_S, input_C_S) * w_cycle
loss_cycle_BAB = F.l1_loss(recon_C_T, input_C_T) * w_cycle
loss_G_adv_A = self.criterionGAN(self.net_D_A(fake_C_T), True) * w_adv
loss_G_adv_B = self.criterionGAN(self.net_D_B(fake_C_S), True) * w_adv
loss_perceptual = (...) * w_percep
loss_c_ssim = (...) * w_ssim
loss_idt = (...) * self.cfg.trainer.idt_weight * w_cycle
```

**关键设计**：
- 精细模式：对抗权重=0（不施加"像Map C"的压力），SSIM权重=15（全结构保持）
- 粗粒模式：对抗权重=3（激进的域转换），SSIM权重=3（弱结构约束）

---

### 文件3: `configs/maps/translation_cgrad.yaml`

在trainer段添加（line 185之后）：

```yaml
  # for scale-disentangled translation (fine mode: structure preserve)
  scale_fine_ssim_weight: 15.0
  scale_fine_adv_weight: 0.0
  scale_fine_cycle_weight: 40
  scale_fine_percep_weight: 20
  # for scale-disentangled translation (coarse mode: domain transfer)
  scale_coarse_ssim_weight: 3.0
  scale_coarse_adv_weight: 3.0
  scale_coarse_cycle_weight: 20
  scale_coarse_percep_weight: 10
```

同时将 `c_ssim_weight` 回调为0（由scale参数替代）：

```yaml
  c_ssim_weight: 0.0             # 由scale_fine/coarse_ssim_weight替代
  g_adv_A_weight: 0.0            # 由scale_fine/coarse_adv_weight替代
  cycle_ABA_weight: 30           # 回退为默认（由scale参数替代）
```

---

## 3. 推理流程

推理时使用`sample_from_c_list`的c_list数据：

```python
# 每个c_list[i]根据其t_i选择对应的scale_id
for i, c_i in enumerate(c_list):
    t_i = t_steps[i+1]  # 对应的时间步
    if t_i < 0.5:
        c_translated[i] = net_G_A(c_i, t_i, scale_id=0)  # 精细: 保守
    else:
        c_translated[i] = net_G_A(c_i, t_i, scale_id=1)  # 粗粒: 激进

c_list_translated → model2.sample_from_c_list() → 像素输出
```

---

## 4. 验证标准

| 指标 | 当前 (统一SSIM, w=15) | 目标 (尺度解耦) | 验证方式 |
|------|--------------------|---------------|---------|
| grad_fake在精细t vs 粗粒t | **≈0.075** (恒定) | **精细t > 0.3, 粗粒t < 0.1** | `verify_scale_hypothesis.py` 重新运行 |
| A2B梯度幅值 | 15.05 | **> 16.0** | `quick_a2b_c_vis.py` |
| 恒等映射余弦 | 0.905 (退化) | **> 0.95** | `evaluate_cyclediff.py --direction A2B` |
| 对抗损失在精细模式 | N/A | **≈ 0** (不施加对抗) | 训练日志 |
| 对抗损失在粗粒模式 | 1.89 | **< 1.0** (专注域转换) | 训练日志 |

**关键验证**：重新运行 `verify_scale_hypothesis.py` → `grad_fake` 在精细t和粗粒t之间应有显著差异。这是尺度解耦**最直接的成功/失败判据**——如果scale_id被generator忽略，则`grad_fake`仍恒定。

---

## 5. 训练配置

| 参数 | 值 | 说明 |
|------|----|------|
| `train_num_steps` | 3000 | 中长度验证 |
| `batch_size` | 20 | 保持，仅×2前向（vs ×3） |
| 每步时间 | ~160ms | (vs 120ms原版, +33%) |
| 总训练时间 | ~2小时 | 可接受 |
| `scale_fine_adv_weight` | 0.0 | 精细层完全不禁对抗 |
| `scale_coarse_adv_weight` | 3.0 | 粗粒层专注域转换 |

---

## 6. 实施步骤

| 步骤 | 内容 | 文件 | 改动行数 |
|------|------|------|---------|
| 1 | 添加scale_id注入到generator | `cycle_generator_2.py` | ~14行 |
| 2 | 采样策略和scale_id传递 | `train_uncond_...py:600,609-623` | ~15行 |
| 3 | 尺度相关损失权重 | `train_uncond_...py:636-668` | ~20行 |
| 4 | 配置参数 | `translation_cgrad.yaml` | ~12行 |
| **合计** | | | **~61行** |

---

## 7. 风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| scale_id被generator忽略 | 低 | FiLM机制已能在一步梯度内切换行为；精细模式的强SSIM约束会迫使generator关注scale_id |
| 精细t和粗粒t的分界值(0.5)不最优 | 低 | 通过config可调，无需代码改动 |
| net_G_B不接收scale_id导致非对称性 | 极低 | B→A方向本无退化问题；net_G_A的scale_id行为通过cycle/identity路径间接约束net_G_B |
| batch随机择尺度导致训练不稳定 | 低 | 50%概率采样的随机性本质上与当前随机t相同——只是多了一个离散维度 |
| 当前统一SSIM权重(15)降低到精细SSIM(15)+粗粒SSIM(3) → 总SSIM约束变弱 | 中 | 精细模式的SSIM=15保持原约束水平；粗粒模式的弱SSIM是设计意图 |
