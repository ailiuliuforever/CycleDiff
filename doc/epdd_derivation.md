# Edge-Preserving Decoupled Diffusion (EPDD) 详细数学推导与可行性分析

## 目录

1. [可行性分析](#1-可行性分析)
2. [正向加噪过程推导](#2-正向加噪过程推导)
3. [反向去噪过程推导](#3-反向去噪过程推导)
4. [训练目标推导](#4-训练目标推导)
5. [完整变量说明表](#5-完整变量说明表)
6. [Python 验证代码](#6-python-验证代码)

***

## 1. 可行性分析

### 1.1 核心问题

解耦扩散模型的正向过程为：

$$
\mathbf{x}\_t = (1-t)\mathbf{x}\_0 + t\boldsymbol{\epsilon}
$$

该过程使用**各向同性噪声**，对所有像素均匀加噪。这导致：

- 边缘像素和平滑像素受到相同程度的噪声干扰
- 图像分量 $C = -\mathbf{x}\_0$ 虽然能恢复干净图像，但去噪网络需要从严重破坏边缘的噪声图像中预测
- 跨域翻译时，结构信息（边缘）的保留不足

### 1.2 边缘保持扩散的优势

边缘保持扩散通过引入基于梯度的非各向同性噪声：

- 边缘处（梯度大）：噪声系数小，保留结构
- 平滑处（梯度小）：噪声系数大，正常加噪

### 1.3 结合的可行性

**关键洞察**：解耦扩散的线性衰减形式 $(1-t)\mathbf{x}\_0$ 与边缘保持噪声系数可以自然结合。

可行性依据：

1. **数学兼容性**：两者都是线性扩散过程，可以叠加
2. **物理合理性**：图像衰减（信号减少）和噪声增长（噪声增加）是两个独立过程，可以分别控制
3. **计算可行性**：梯度计算开销小，现代 GPU 可高效处理
4. **收敛保证**：通过混合方案（早期边缘保持 + 晚期各向同性），确保最终收敛到高斯分布

### 1.4 预期优势

| 方面        | 原始解耦扩散    | 边缘保持解耦扩散      |
| --------- | --------- | ------------- |
| 边缘保留      | 差（各向同性破坏） | 好（边缘处噪声小）     |
| 图像分量质量    | 一般        | 更高（结构更清晰）     |
| 跨域翻译结构一致性 | 一般        | 更好            |
| 计算复杂度     | O(HWC)    | O(HWC) + 梯度计算 |

***

## 2. 正向加噪过程推导

### 2.1 基础形式

借鉴解耦扩散的线性衰减和边缘保持扩散的非各向同性噪声，定义正向过程：

$$
\mathbf{x}\_t = (1-t)\mathbf{x}\_0 + t \cdot \boldsymbol{\sigma}\_t^{\text{EP}} \cdot \boldsymbol{\epsilon} \tag{EPDD-1}
$$

其中：

- $(1-t)\mathbf{x}\_0$：图像线性衰减项（与解耦扩散相同）
- $t \cdot \boldsymbol{\sigma}\_t^{\text{EP}} \cdot \boldsymbol{\epsilon}$：边缘保持噪声增长项
- $\boldsymbol{\sigma}\_t^{\text{EP}}$：边缘保持噪声系数（张量，逐像素不同）

### 2.2 边缘保持噪声系数

受 Perona-Malik 扩散系数启发，定义：

$$
\boldsymbol{\sigma}\_t^{\text{EP}} = \frac{1}{(1 - \tau(t))\sqrt{1 + \frac{|\nabla \mathbf{x}\_0|}{\lambda(t)}} + \tau(t)} \tag{EPDD-2}
$$

其中：

- $|\nabla \mathbf{x}\_0|$：原始图像的梯度幅值
- $\lambda(t)$：时变边缘敏感度
- $\tau(t)$：过渡函数

**物理意义**：

- 边缘处（$|\nabla \mathbf{x}\_0| \gg \lambda(t)$）：$\sqrt{1 + \frac{|\nabla \mathbf{x}\_0|}{\lambda(t)}} \gg 1$，分母主要由 $(1-\tau(t))\sqrt{1 + \frac{|\nabla \mathbf{x}\_0|}{\lambda(t)}}$ 主导，因此 $\sigma\_t^{\text{EP}} \approx \frac{1}{(1-\tau(t))\sqrt{1 + \frac{|\nabla \mathbf{x}\_0|}{\lambda(t)}}} \ll 1$（噪声系数小，边缘处注入噪声少）
- 平滑处（$|\nabla \mathbf{x}\_0| \ll \lambda(t)$）：$\sqrt{1 + \frac{|\nabla \mathbf{x}\_0|}{\lambda(t)}} \approx 1$，因此 $\sigma\_t^{\text{EP}} \approx \frac{1}{(1-\tau(t)) + \tau(t)} = 1$（噪声系数退化为标准值）

**注意**：当 $\tau(t) = 1$ 时，无论梯度大小，$\sigma\_t^{\text{EP}} = 1$，退化为各向同性噪声。

### 2.3 混合噪声方案

**过渡函数** $\tau(t)$：

- 当 $t < t\_\Phi$（过渡点）：$\tau(t) < 1$，边缘保持生效
- 当 $t \geq t\_\Phi$：$\tau(t) = 1$，退化为各向同性

线性过渡函数示例：

$$
\tau(t) = \begin{cases}
\frac{t}{t\_\Phi} & t < t\_\Phi \\
1 & t \geq t\_\Phi
\end{cases}
$$

### 2.4 时变边缘敏感度

$$
\lambda(t) = \lambda\_{\min} + t(\lambda\_{\max} - \lambda\_{\min})
$$

- 早期（$t$ 小）：$\lambda(t) \approx \lambda\_{\min}$，强边缘保持
- 晚期（$t$ 大）：$\lambda(t) \approx \lambda\_{\max}$，弱边缘保持

### 2.5 边界条件验证

**当 $t = 0$ 时**：

$$
\mathbf{x}\_0 = (1-0)\mathbf{x}\_0 + 0 \cdot \boldsymbol{\sigma}\_0^{\text{EP}} \cdot \boldsymbol{\epsilon} = \mathbf{x}\_0 \quad \checkmark
$$

**当 $t = 1$ 时**：

$$
\mathbf{x}\_1 = (1-1)\mathbf{x}\_0 + 1 \cdot \boldsymbol{\sigma}\_1^{\text{EP}} \cdot \boldsymbol{\epsilon} = \boldsymbol{\sigma}\_1^{\text{EP}} \cdot \boldsymbol{\epsilon}
$$

由于 $t = 1 \geq t\_\Phi$，$\tau(1) = 1$，所以 $\boldsymbol{\sigma}\_1^{\text{EP}} = 1$：

$$
\mathbf{x}\_1 = \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \checkmark
$$

### 2.6 图像分量

与解耦扩散类似，图像分量为：

$$
C = -\mathbf{x}\_0
$$

但此时从 $\mathbf{x}\_t$ 中恢复 $\mathbf{x}\_0$ 需要知道 $\boldsymbol{\sigma}\_t^{\text{EP}}$：

$$
\mathbf{x}\_0 = \frac{\mathbf{x}\_t - t \cdot \boldsymbol{\sigma}\_t^{\text{EP}} \cdot \boldsymbol{\epsilon}}{1-t}
$$

***

## 3. 反向去噪过程推导

### 3.1 去噪网络设计

与解耦扩散类似，去噪网络同时输出：

$$
C\_\theta, \boldsymbol{\epsilon}_\theta = \operatorname{Net}_\theta(\mathbf{x}\_t, t) \tag{EPDD-3}
$$

其中：

- $C\_\theta$：预测的图像分量（应接近 $-\mathbf{x}\_0$）
- $\boldsymbol{\epsilon}\_\theta$：预测的噪声

### 3.2 从预测恢复干净图像

如果网络完美预测：

$$
C\_\theta \approx C = -\mathbf{x}\_0
$$

则：

$$
\mathbf{x}_0 \approx -C_\theta \tag{EPDD-4}
$$

### 3.3 反向去噪更新公式（基于 ODE 解析解）

从正向过程（EPDD-1），对时间 $t$ 求导。由于 $\boldsymbol{\sigma}_t^{\text{EP}}$ 通过 $\tau(t)$ 和 $\lambda(t)$ 依赖于 $t$，需要应用乘积法则：

$$
\frac{d\mathbf{x}\_t}{dt} = -\mathbf{x}\_0 + \frac{d}{dt}\left[t \cdot \boldsymbol{\sigma}\_t^{\text{EP}}\right] \cdot \boldsymbol{\epsilon} = -\mathbf{x}\_0 + \left(\boldsymbol{\sigma}\_t^{\text{EP}} + t \cdot \frac{d\boldsymbol{\sigma}\_t^{\text{EP}}}{dt}\right) \cdot \boldsymbol{\epsilon}
$$

定义**有效噪声系数**：

$$
\tilde{\sigma}(t) = \boldsymbol{\sigma}\_t^{\text{EP}} + t \cdot \frac{d\boldsymbol{\sigma}\_t^{\text{EP}}}{dt} \tag{EPDD-5a}
$$

利用 $C = -\mathbf{x}\_0$，严格的 ODE 形式为：

$$
\frac{d\mathbf{x}\_t}{dt} = C + \tilde{\sigma}(t) \cdot \boldsymbol{\epsilon} \tag{EPDD-5b}
$$

在反向去噪过程中，网络预测 $C_\theta$ 和 $\boldsymbol{\epsilon}\_\theta$，因此 ODE 的近似形式为：

$$
\frac{d\mathbf{x}\_t}{dt} \approx C_\theta + \tilde{\sigma}(t) \cdot \boldsymbol{\epsilon}_\theta
$$

对时间步长 $s$ 进行欧拉积分，得到反向去噪更新公式：

$$
\mathbf{x}\_{t-s} = \mathbf{x}_t - s \cdot \left(C_\theta + \tilde{\sigma}(t) \cdot \boldsymbol{\epsilon}_\theta\right) \tag{EPDD-5}
$$

**$\boldsymbol{\sigma}\_t^{\text{EP}}$ 的导数计算**：

令 $D(t) = (1-\tau(t))\cdot A(t) + \tau(t)$，其中 $A(t) = \sqrt{1 + |\nabla \mathbf{x}\_0|/\lambda(t)}$。

则 $\boldsymbol{\sigma}\_t^{\text{EP}} = 1/D(t)$，且：

$$
\frac{d\boldsymbol{\sigma}\_t^{\text{EP}}}{dt} = -\frac{1}{D(t)^2} \cdot \frac{dD}{dt}
$$

其中：

$$
\frac{dD}{dt} = \frac{d\tau}{dt} \cdot (1 - A(t)) + (1 - \tau(t)) \cdot \frac{dA}{dt}
$$

而：

$$
\frac{dA}{dt} = \frac{-|\nabla \mathbf{x}\_0| \cdot (\lambda\_{\max} - \lambda\_{\min})}{2 \cdot \lambda(t)^2 \cdot A(t)}
$$

对于 $d\tau/dt$：

- 当 $t < t\_\Phi$ 时：$d\tau/dt = 1/t\_\Phi$
- 当 $t \geq t\_\Phi$ 时：$d\tau/dt = 0$

**重要性质**：

- 当 $t \geq t\_\Phi$ 时：$\tau(t)=1$，$D(t)=1$，$\sigma\_t^{\text{EP}}=1$，$d\sigma/dt=0$，因此 $\tilde{\sigma}(t)=1$（退化为标准扩散）
- 当 $t < t\_\Phi$ 时：修正项 $t \cdot d\sigma/dt$ 可达 $\sigma\_t^{\text{EP}}$ 的 60%~138%（数值验证），不可忽略

**近似形式（单步冻结近似）**：

若假设 $\boldsymbol{\sigma}\_t^{\text{EP}}$ 在单个积分步内变化缓慢（即 $d\boldsymbol{\sigma}\_t^{\text{EP}}/dt \approx 0$），则 $\tilde{\sigma}(t) \approx \boldsymbol{\sigma}\_t^{\text{EP}}$，更新退化为：

$$
\mathbf{x}\_{t-s} \approx \mathbf{x}_t - s \cdot (C_\theta + \boldsymbol{\sigma}\_t^{\text{EP}} \cdot \boldsymbol{\epsilon}_\theta)
$$

然而，数值验证表明该近似在边缘区域的 MAE 可达 0.02~0.09，因此推荐使用严格形式。

**物理意义**：

- $C\_\theta$：图像分量更新项，负责恢复干净图像结构
- $\tilde{\sigma}(t) \cdot \boldsymbol{\epsilon}_\theta$：噪声更新项，受有效噪声系数调制
- 边缘处（$\tilde{\sigma}(t)$ 小）：噪声更新项小，去噪更确定性，保留边缘结构
- 平滑处（$\tilde{\sigma}(t)$ 大）：噪声更新项大，正常去噪

### 3.4 与原始解耦扩散 ODE 的对比

| 特性     | 原始解耦扩散 ODE                                                                   | EPDD ODE                                                                                                                |
| ------ | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 正向过程   | $\mathbf{x}\_t = (1-t)\mathbf{x}\_0 + t\boldsymbol{\epsilon}$                | $\mathbf{x}\_t = (1-t)\mathbf{x}\_0 + t \cdot \boldsymbol{\sigma}\_t^{\text{EP}} \cdot \boldsymbol{\epsilon}$           |
| ODE 形式 | $\frac{d\mathbf{x}\_t}{dt} = C + \boldsymbol{\epsilon}$                      | $\frac{d\mathbf{x}\_t}{dt} = C + \tilde{\sigma}(t) \cdot \boldsymbol{\epsilon}$                                        |
| 反向更新   | $\mathbf{x}_{t-s} = \mathbf{x}t - s(C_\theta + \boldsymbol{\epsilon}\theta)$ | $\mathbf{x}\_{t-s} = \mathbf{x}_t - s(C_\theta + \tilde{\sigma}(t) \cdot \boldsymbol{\epsilon}_\theta)$ |
| 噪声系数   | 标量 1                                                                         | 张量 $\tilde{\sigma}(t) = \boldsymbol{\sigma}\_t^{\text{EP}} + t \cdot \frac{d\boldsymbol{\sigma}\_t^{\text{EP}}}{dt}$（逐像素不同） |
| 边缘处理   | 均匀更新                                                                         | 边缘处更新小，平滑处更新大                                                                                                           |

***

## 4. 训练目标推导

### 4.1 扩散模型损失

网络预测 $C_\theta$ 和 $\boldsymbol{\epsilon}_\theta$。ODE 速度场为 $C + \tilde{\sigma}(t) \cdot \boldsymbol{\epsilon}$。网络学习预测 $C$ 和 $\boldsymbol{\epsilon}$，使得 $C_\theta \approx C$ 且 $\boldsymbol{\epsilon}_\theta \approx \boldsymbol{\epsilon}$。采样时 ODE 速度通过 $C_\theta + \tilde{\sigma}(t) \cdot \boldsymbol{\epsilon}_\theta$ 重构。

因此训练目标为同时监督图像分量和噪声：

$$
\mathcal{L}_{dm} = \mathbb{E}\left\[|C_\theta - C|^2 + |\boldsymbol{\epsilon}\_\theta - \boldsymbol{\epsilon}|^2\right] \tag{EPDD-6}
$$

其中：

- 第一项：图像分量预测损失，监督网络正确预测 $C = -\mathbf{x}\_0$
- 第二项：噪声预测损失，监督网络正确预测噪声 $\boldsymbol{\epsilon}$

**关键说明**：

1. $C_\theta$ 直接预测图像分量 $C = -\mathbf{x}\_0$
2. $\boldsymbol{\epsilon}_\theta$ 直接预测噪声 $\boldsymbol{\epsilon}$
3. ODE 速度在采样时通过 $C_\theta + \tilde{\sigma}(t) \cdot \boldsymbol{\epsilon}_\theta$ 重构
4. $\tilde{\sigma}(t)$ 因子在采样阶段施加，而非训练阶段，因此训练目标形式不变

### 4.2 边缘保持加权噪声损失（可选）

为了进一步增强边缘保持效果，可以对噪声损失引入边缘感知权重：

$$
|\boldsymbol{\epsilon}_\theta - \boldsymbol{\epsilon}|^2_{\boldsymbol{\Sigma}} = \sum\_{i,j} \frac{1}{(\sigma\_{t,ij}^{\text{EP}})^2}(\epsilon\_{\theta,ij} - \epsilon\_{ij})^2
$$

此时完整损失为：

$$
\mathcal{L}_{dm}^{\text{weighted}} = \mathbb{E}\left\[|C_\theta - C|^2 + |\boldsymbol{\epsilon}_\theta - \boldsymbol{\epsilon}|^2_{\boldsymbol{\Sigma}}\right]
$$

**物理意义**：

- 边缘处（$\sigma\_t^{\text{EP}}$ 小）：权重 $1/(\sigma\_t^{\text{EP}})^2$ 更大，要求更精确的噪声预测
- 平滑处（$\sigma\_t^{\text{EP}}$ 大，接近 1）：权重 $1/(\sigma\_t^{\text{EP}})^2$ 更小，允许更大的预测误差

**重要说明**：

- 加权损失是**可选的**附加组件，非必须
- 不加权时（EPDD-6），边缘保持效果完全由正向过程的噪声系数 $\boldsymbol{\sigma}\_t^{\text{EP}}$ 实现
- 加权方案可能导致训练不稳定——边缘处的高权重会放大少量噪声的预测误差
- 实际实现中可考虑使用**截断权重** $\min(1/(\sigma\_t^{\text{EP}})^2, w\_{\max})$ 或直接使用不加权方案

### 4.3 与原始解耦扩散的对比

| 方面       | 原始解耦扩散                                                        | EPDD                                                                               |
| -------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 图像分量损失   | $\|C\_\theta - C\|^2$                                         | 相同                                                                                 |
| 噪声损失     | $\|\boldsymbol{\epsilon}\_\theta - \boldsymbol{\epsilon}\|^2$ | 相同（基础形式）                                                                           |
| 可选加权噪声损失 | 无                                                             | $\|\boldsymbol{\epsilon}_\theta - \boldsymbol{\epsilon}\|^2_{\boldsymbol{\Sigma}}$ |
| 网络输出     | 双输出头                                                          | 相同                                                                                 |
| 额外输入     | 无                                                             | 需要 $\|\nabla \mathbf{x}\_0\|$（仅正向过程）                                               |

***

## 5. 完整变量说明表

### 5.1 核心变量

| 变量                                   | 类型   | 数学定义                                                             | 物理意义               |
| ------------------------------------ | ---- | ---------------------------------------------------------------- | ------------------ |
| $\mathbf{x}\_0$                      | 图像张量 | $\mathbf{x}\_0 \in \mathbb{R}^{H \times W \times C}$             | 原始干净图像             |
| $\mathbf{x}\_t$                      | 图像张量 | 公式 (EPDD-1)                                                      | 时刻 $t$ 的加噪图像       |
| $C$                                  | 图像分量 | $C = -\mathbf{x}\_0$                                             | 图像分量（ODE 速度场的常数部分） |
| $\boldsymbol{\epsilon}$              | 随机噪声 | $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | 标准高斯噪声             |
| $\boldsymbol{\sigma}\_t^{\text{EP}}$ | 张量   | 公式 (EPDD-2)                                                      | 边缘保持噪声系数           |

### 5.2 边缘保持相关变量

| 变量                         | 类型   | 数学定义                                                                   | 物理意义    |
| -------------------------- | ---- | ---------------------------------------------------------------------- | ------- |
| $\|\nabla \mathbf{x}\_0\|$ | 梯度幅值 | $\sqrt{(\partial\_x \mathbf{x}\_0)^2 + (\partial\_y \mathbf{x}\_0)^2}$ | 图像边缘强度  |
| $\lambda(t)$               | 标量函数 | $\lambda\_{\min} + t(\lambda\_{\max} - \lambda\_{\min})$               | 时变边缘敏感度 |
| $\tau(t)$                  | 标量函数 | $\[0, T] \to \[0, 1]$                                                  | 过渡函数    |
| $t\_\Phi$                  | 标量   | $0.5T$（典型值）                                                            | 过渡点     |

### 5.3 网络与损失变量

| 变量                              | 类型   | 数学定义                                       | 物理意义       |
| ------------------------------- | ---- | ------------------------------------------ | ---------- |
| $C\_\theta$                     | 预测分量 | $\operatorname{Net}\_\theta\[0]$           | 预测的图像分量    |
| $\boldsymbol{\epsilon}\_\theta$ | 预测噪声 | $\operatorname{Net}\_\theta\[1]$           | 预测的噪声      |
| $\boldsymbol{\Sigma}$           | 权重矩阵 | $\text{diag}(1/(\sigma\_t^{\text{EP}})^2)$ | 边缘保持权重（可选） |
| $\mathcal{L}\_{dm}$             | 损失   | 公式 (EPDD-6)                                | 扩散模型损失     |

### 5.4 ODE 相关变量

| 变量                          | 类型   | 数学定义                                                                         | 物理意义      |
| --------------------------- | ---- | ------------------------------------------------------------------------------ | --------- |
| $\frac{d\mathbf{x}\_t}{dt}$ | 速度场  | $C + \tilde{\sigma}(t) \cdot \boldsymbol{\epsilon}$                             | ODE 速度场   |
| $\tilde{\sigma}(t)$         | 张量   | $\boldsymbol{\sigma}\_t^{\text{EP}} + t \cdot \frac{d\boldsymbol{\sigma}\_t^{\text{EP}}}{dt}$（EPDD-5a） | 有效噪声系数    |
| $s$                         | 标量   | 时间步长                                                                           | 反向去噪步长    |
| $\mathbf{x}\_{t-s}$         | 图像张量 | 公式 (EPDD-5)                                                                    | 去噪后的图像    |

***

## 6. Python 验证代码

```python
import torch
import torch.nn.functional as F
import math


def compute_gradient_magnitude(x):
    """计算图像梯度幅值"""
    # x: (B, C, H, W)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(x, sobel_x.repeat(x.shape[1], 1, 1, 1), 
                      padding=1, groups=x.shape[1])
    grad_y = F.conv2d(x, sobel_y.repeat(x.shape[1], 1, 1, 1), 
                      padding=1, groups=x.shape[1])
    
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
    return grad_mag


def edge_preserving_sigma(x0, t, lambda_min=0.1, lambda_max=10.0, 
                          t_phi=0.5):
    """
    计算边缘保持噪声系数 sigma_t^EP (EPDD-2)
    
    Args:
        x0: 原始图像 (B, C, H, W)
        t: 时间步 (B,)
        lambda_min: 最小边缘敏感度
        lambda_max: 最大边缘敏感度
        t_phi: 过渡点
    
    Returns:
        sigma: (B, 1, H, W) 边缘保持噪声系数
    """
    grad_mag = compute_gradient_magnitude(x0)
    grad_mag = grad_mag.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    lambda_t = lambda_min + t * (lambda_max - lambda_min)
    lambda_t = lambda_t.view(-1, 1, 1, 1)
    
    tau = torch.where(t < t_phi, t / t_phi, torch.ones_like(t))
    tau = tau.view(-1, 1, 1, 1)
    
    denominator = (1 - tau) * torch.sqrt(1 + grad_mag / lambda_t) + tau
    sigma = 1.0 / denominator
    
    return sigma


def compute_effective_sigma(x0, t, lambda_min=0.1, lambda_max=10.0, 
                            t_phi=0.5):
    """
    计算有效噪声系数 sigma_tilde(t) = sigma_t^EP + t * d(sigma_t^EP)/dt (EPDD-5a)
    
    Args:
        x0: 原始图像 (B, C, H, W)
        t: 时间步 (B,)
        lambda_min: 最小边缘敏感度
        lambda_max: 最大边缘敏感度
        t_phi: 过渡点
    
    Returns:
        sigma_tilde: (B, 1, H, W) 有效噪声系数
        sigma_ep: (B, 1, H, W) 边缘保持噪声系数
        dsigma_dt: (B, 1, H, W) sigma_t^EP 对 t 的导数
    """
    grad_mag = compute_gradient_magnitude(x0)
    grad_mag = grad_mag.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    lambda_t = lambda_min + t * (lambda_max - lambda_min)
    lambda_t = lambda_t.view(-1, 1, 1, 1)
    
    tau = torch.where(t < t_phi, t / t_phi, torch.ones_like(t))
    tau = tau.view(-1, 1, 1, 1)
    
    A = torch.sqrt(1 + grad_mag / lambda_t)
    D = (1 - tau) * A + tau
    sigma_ep = 1.0 / D
    
    dtau_dt = torch.where(t < t_phi, 
                          torch.full_like(t, 1.0 / t_phi), 
                          torch.zeros_like(t))
    dtau_dt = dtau_dt.view(-1, 1, 1, 1)
    
    dA_dt = -grad_mag * (lambda_max - lambda_min) / (2 * lambda_t**2 * A)
    
    dD_dt = dtau_dt * (1 - A) + (1 - tau) * dA_dt
    
    dsigma_dt = -1.0 / (D**2) * dD_dt
    
    t_expanded = t.view(-1, 1, 1, 1)
    sigma_tilde = sigma_ep + t_expanded * dsigma_dt
    
    return sigma_tilde, sigma_ep, dsigma_dt


def forward_process(x0, t, sigma_ep=None):
    """
    EPDD 正向加噪过程 (EPDD-1)
    
    Args:
        x0: 原始图像 (B, C, H, W)
        t: 时间步 (B,)
        sigma_ep: 边缘保持噪声系数 (B, 1, H, W)，若为 None 则计算
    
    Returns:
        xt: 加噪图像 (B, C, H, W)
        noise: 使用的噪声 (B, C, H, W)
    """
    if sigma_ep is None:
        sigma_ep = edge_preserving_sigma(x0, t)
    
    noise = torch.randn_like(x0)
    
    t = t.view(-1, 1, 1, 1)
    xt = (1 - t) * x0 + t * sigma_ep * noise
    
    return xt, noise


def reverse_denoise_step(xt, t, s, C_theta, eps_theta, sigma_tilde):
    """
    EPDD 反向去噪单步更新 (EPDD-5)
    
    Args:
        xt: 当前加噪图像 (B, C, H, W)
        t: 当前时间步 (B,)
        s: 步长 (B,)
        C_theta: 预测的图像分量 (B, C, H, W)
        eps_theta: 预测的噪声 (B, C, H, W)
        sigma_tilde: 有效噪声系数 (B, 1, H, W)
    
    Returns:
        xt_prev: 去噪后的图像 (B, C, H, W)
    """
    s = s.view(-1, 1, 1, 1)
    
    xt_prev = xt - s * (C_theta + sigma_tilde * eps_theta)
    
    return xt_prev


def training_loss(C_theta, eps_theta, x0, noise, sigma_ep=None, 
                  use_weighting=False):
    """
    EPDD 训练损失 (EPDD-6)
    
    Args:
        C_theta: 预测的图像分量 (B, C, H, W)
        eps_theta: 预测的噪声 (B, C, H, W)
        x0: 原始图像 (B, C, H, W)
        noise: 真实噪声 (B, C, H, W)
        sigma_ep: 边缘保持噪声系数 (B, 1, H, W)
        use_weighting: 是否使用边缘加权
    
    Returns:
        loss: 标量损失
    """
    C = -x0
    
    loss_C = torch.mean((C_theta - C)**2)
    
    if use_weighting and sigma_ep is not None:
        weights = 1.0 / (sigma_ep**2)
        weights = torch.clamp(weights, max=100.0)
        loss_eps = torch.mean(weights * (eps_theta - noise)**2)
    else:
        loss_eps = torch.mean((eps_theta - noise)**2)
    
    loss = loss_C + loss_eps
    return loss


# ==================== 验证测试 ====================
if __name__ == "__main__":
    B, C, H, W = 2, 3, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x0 = torch.randn(B, C, H, W, device=device)
    t = torch.rand(B, device=device) * 0.9 + 0.1  # 避免 t=0
    
    print("=" * 50)
    print("EPDD 验证测试")
    print("=" * 50)
    
    # 1. 测试边缘保持噪声系数
    sigma_ep = edge_preserving_sigma(x0, t)
    print(f"\n1. 边缘保持噪声系数:")
    print(f"   sigma_ep 形状: {sigma_ep.shape}")
    print(f"   sigma_ep 范围: [{sigma_ep.min():.4f}, {sigma_ep.max():.4f}]")
    print(f"   sigma_ep 均值: {sigma_ep.mean():.4f}")
    
    # 2. 测试有效噪声系数
    sigma_tilde, sigma_ep_2, dsigma_dt = compute_effective_sigma(x0, t)
    print(f"\n2. 有效噪声系数 (EPDD-5a):")
    print(f"   sigma_tilde 形状: {sigma_tilde.shape}")
    print(f"   sigma_tilde 范围: [{sigma_tilde.min():.4f}, {sigma_tilde.max():.4f}]")
    print(f"   sigma_tilde 均值: {sigma_tilde.mean():.4f}")
    correction_ratio = (t.view(-1, 1, 1, 1) * dsigma_dt).abs() / sigma_ep_2.abs().clamp(min=1e-8)
    print(f"   修正项 t·dσ/dt 占 σ_EP 的比例: [{correction_ratio.min():.4f}, {correction_ratio.max():.4f}]")
    
    # 3. 测试正向过程
    xt, noise = forward_process(x0, t, sigma_ep)
    print(f"\n3. 正向加噪过程 (EPDD-1):")
    print(f"   x0 范围: [{x0.min():.4f}, {x0.max():.4f}]")
    print(f"   xt 范围: [{xt.min():.4f}, {xt.max():.4f}]")
    
    t0 = torch.zeros(B, device=device)
    xt0, _ = forward_process(x0, t0, edge_preserving_sigma(x0, t0))
    print(f"   t=0 时 xt≈x0: {torch.allclose(xt0, x0, atol=1e-5)}")
    
    # 4. 测试反向去噪（使用严格形式）
    C_theta = -x0 + 0.1 * torch.randn_like(x0)
    eps_theta = noise + 0.1 * torch.randn_like(noise)
    
    s = torch.full((B,), 0.1, device=device)
    sigma_tilde_val, _, _ = compute_effective_sigma(x0, t)
    xt_prev = reverse_denoise_step(xt, t, s, C_theta, eps_theta, sigma_tilde_val)
    print(f"\n4. 反向去噪更新 (EPDD-5 严格形式):")
    print(f"   xt 范围: [{xt.min():.4f}, {xt.max():.4f}]")
    print(f"   xt_prev 范围: [{xt_prev.min():.4f}, {xt_prev.max():.4f}]")
    
    # 5. 对比严格形式与近似形式
    sigma_ep_val = edge_preserving_sigma(x0, t)
    xt_prev_approx = xt - s.view(-1, 1, 1, 1) * (C_theta + sigma_ep_val * eps_theta)
    mae = (xt_prev - xt_prev_approx).abs().mean()
    print(f"\n5. 严格形式 vs 近似形式对比:")
    print(f"   MAE: {mae.item():.6f}")
    print(f"   最大绝对误差: {(xt_prev - xt_prev_approx).abs().max().item():.6f}")
    
    # 6. 测试训练损失
    loss = training_loss(C_theta, eps_theta, x0, noise, sigma_ep, 
                         use_weighting=False)
    loss_weighted = training_loss(C_theta, eps_theta, x0, noise, sigma_ep, 
                                  use_weighting=True)
    print(f"\n6. 训练损失 (EPDD-6):")
    print(f"   标准损失: {loss.item():.6f}")
    print(f"   加权损失: {loss_weighted.item():.6f}")
    
    # 7. 验证边界条件
    print(f"\n7. 边界条件验证:")
    t1 = torch.ones(B, device=device)
    sigma_tilde_1, sigma_ep_1, dsigma_dt_1 = compute_effective_sigma(x0, t1)
    print(f"   t=1 时 sigma_ep≈1: {torch.allclose(sigma_ep_1, torch.ones_like(sigma_ep_1), atol=1e-5)}")
    print(f"   t=1 时 sigma_tilde≈1: {torch.allclose(sigma_tilde_1, torch.ones_like(sigma_tilde_1), atol=1e-5)}")
    print(f"   t=1 时 dσ/dt≈0: {torch.allclose(dsigma_dt_1, torch.zeros_like(dsigma_dt_1), atol=1e-5)}")
    
    print("\n" + "=" * 50)
    print("所有验证完成!")
    print("=" * 50)
```

