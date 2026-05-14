"""
验证 compute_effective_noise_coefficient 产生正确的 ODE 速度

核心数学:
  EPDD 前向: x_t = (1-t)*x_0 + t*sigma_ep*epsilon
  ODE 速度:  dx_t/dt = -x_0 + sigma_ep*epsilon + t*(d_sigma_ep/dt)*epsilon
                    = C + sigma_eff*epsilon
  其中 sigma_eff = sigma_ep + t * d_sigma_ep/dt

  旧(错误)形式: C + sigma_ep * epsilon  (遗漏 t*d_sigma_ep/dt 项)
  新(正确)形式: C + sigma_eff * epsilon (使用 compute_effective_noise_coefficient)
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/root/autodl-tmp/CycleDiff')

from ddm.ddm_const_ode_2_learn import EdgeLatentDiffusion
from ddm.encoder_decoder import AutoencoderKL


class DummyUnet(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.channels = channels
        self.self_condition = False
        self.C_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.noise_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, t, cond=None):
        B, C, H, W = x.shape
        C_pred = self.C_bias.expand(B, C, H, W)
        noise_pred = self.noise_bias.expand(B, C, H, W)
        return C_pred, noise_pred


def create_epdd(image_size=(64, 64), z_channels=4):
    ddconfig = {
        'double_z': True,
        'z_channels': z_channels,
        'resolution': image_size,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 32,
        'ch_mult': [1, 2],
        'num_res_blocks': 1,
        'attn_resolutions': [],
        'dropout': 0.0
    }
    lossconfig = {'disc_start': 50001, 'kl_weight': 0.000001, 'disc_weight': 0.5}

    auto_encoder = AutoencoderKL(ddconfig, lossconfig, embed_dim=z_channels)
    unet = DummyUnet(channels=z_channels)

    cfg = {
        'eps': 1e-4,
        'sigma_min': 1e-2,
        'sigma_max': 1,
        'weighting_loss': False,
        'sample_type': 'deterministic',
        'use_disloss': False,
    }

    epdd = EdgeLatentDiffusion(
        auto_encoder=auto_encoder,
        model=unet,
        image_size=image_size,
        scale_factor=1.0,
        scale_by_std=False,
        cfg=cfg,
        sampling_timesteps=10,
        epdd_lambda_min=1e-5,
        epdd_lambda_max=1e-1,
        epdd_transition_point=0.5,
        use_edge_weighted_loss=False
    )
    return epdd


def create_step_image(B=1, C=4, H=32, W=32):
    z = torch.zeros(B, C, H, W)
    z[:, :, :, W // 2:] = 1.0
    return z


def main():
    torch.manual_seed(42)

    epdd = create_epdd()
    epdd.eval()

    B, C, H, W = 1, 4, 32, 32
    x0 = create_step_image(B, C, H, W)
    C_vec = -x0

    noise = torch.randn(B, C, H, W)

    t_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    dt = 1e-6

    print("=" * 100)
    print("  验证 compute_effective_noise_coefficient 产生正确的 ODE 速度")
    print("=" * 100)
    print()
    print(f"  前向过程: x_t = (1-t)*x_0 + t*sigma_ep*epsilon")
    print(f"  ODE 速度: dx_t/dt = C + sigma_eff*epsilon")
    print(f"  sigma_eff = sigma_ep + t * d(sigma_ep)/dt  (compute_effective_noise_coefficient)")
    print(f"  dt (数值微分步长) = {dt}")
    print(f"  t_Phi (过渡点) = {epdd.epdd_transition_point}")
    print()

    header = f"{'t':>6s} | {'Numerical MAE':>14s} | {'Document MAE':>14s} | {'Strict MAE':>14s} | {'Doc/Strict':>10s} | {'sigma_ep(edge)':>14s} | {'sigma_eff(edge)':>14s}"
    print(header)
    print("-" * len(header))

    for t_val in t_values:
        t = torch.tensor([t_val])

        sigma_ep = epdd.compute_epdd_noise_coefficient(x0, t)
        sigma_eff = epdd.compute_effective_noise_coefficient(x0, t)

        x_t = epdd.q_sample(x0, noise, t, C_vec, sigma_ep=sigma_ep)

        t_plus = torch.tensor([t_val + dt])
        sigma_ep_plus = epdd.compute_epdd_noise_coefficient(x0, t_plus)
        x_t_plus = epdd.q_sample(x0, noise, t_plus, C_vec, sigma_ep=sigma_ep_plus)

        numerical_deriv = (x_t_plus - x_t) / dt

        time_expanded = t.reshape(-1, 1, 1, 1)
        doc_velocity = C_vec + sigma_ep * noise
        strict_velocity = C_vec + sigma_eff * noise

        mae_numerical = numerical_deriv.abs().mean().item()
        mae_doc = (doc_velocity - numerical_deriv).abs().mean().item()
        mae_strict = (strict_velocity - numerical_deriv).abs().mean().item()

        edge_col = W // 2
        edge_sigma_ep = sigma_ep[0, 0, :, edge_col].mean().item()
        edge_sigma_eff = sigma_eff[0, 0, :, edge_col].mean().item()

        ratio = mae_doc / mae_strict if mae_strict > 0 else float('inf')

        print(f"{t_val:6.2f} | {mae_numerical:14.6e} | {mae_doc:14.6e} | {mae_strict:14.6e} | {ratio:10.2f} | {edge_sigma_ep:14.6f} | {edge_sigma_eff:14.6f}")

    print()
    print("=" * 100)
    print("  验证 1: t >= t_Phi 时, sigma_eff ≈ 1.0 (因为 d_sigma/dt = 0 当 tau=1)")
    print("=" * 100)

    for t_val in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        t = torch.tensor([t_val])
        sigma_eff = epdd.compute_effective_noise_coefficient(x0, t)
        sigma_ep = epdd.compute_epdd_noise_coefficient(x0, t)

        flat_sigma_eff = sigma_eff.mean().item()
        flat_sigma_ep = sigma_ep.mean().item()
        edge_col = W // 2
        edge_sigma_eff = sigma_eff[0, 0, :, edge_col].mean().item()
        edge_sigma_ep = sigma_ep[0, 0, :, edge_col].mean().item()

        print(f"  t={t_val:.1f}: sigma_ep(flat)={flat_sigma_ep:.6f}, sigma_eff(flat)={flat_sigma_eff:.6f}, "
              f"sigma_ep(edge)={edge_sigma_ep:.6f}, sigma_eff(edge)={edge_sigma_eff:.6f}")

    print()
    print("=" * 100)
    print("  验证 2: t < t_Phi 时, sigma_eff != sigma_ep 在边缘区域")
    print("=" * 100)

    for t_val in [0.05, 0.1, 0.2, 0.3, 0.4]:
        t = torch.tensor([t_val])
        sigma_ep = epdd.compute_epdd_noise_coefficient(x0, t)
        sigma_eff = epdd.compute_effective_noise_coefficient(x0, t)

        edge_col = W // 2
        edge_sigma_ep = sigma_ep[0, 0, :, edge_col].mean().item()
        edge_sigma_eff = sigma_eff[0, 0, :, edge_col].mean().item()
        diff = abs(edge_sigma_eff - edge_sigma_ep)

        flat_col = 4
        flat_sigma_ep = sigma_ep[0, 0, :, flat_col].mean().item()
        flat_sigma_eff = sigma_eff[0, 0, :, flat_col].mean().item()
        flat_diff = abs(flat_sigma_eff - flat_sigma_ep)

        print(f"  t={t_val:.2f}: edge |sigma_eff-sigma_ep|={diff:.6f}, flat |sigma_eff-sigma_ep|={flat_diff:.6f}, "
              f"edge_sigma_ep={edge_sigma_ep:.6f}, edge_sigma_eff={edge_sigma_eff:.6f}")

    print()
    print("=" * 100)
    print("  总结: Strict ODE 速度 (使用 sigma_eff) 与数值导数的 MAE 对比")
    print("=" * 100)

    print()
    print(f"  {'t':>6s} | {'Document MAE':>14s} | {'Strict MAE':>14s} | {'Improvement':>12s}")
    print("  " + "-" * 60)

    for t_val in t_values:
        t = torch.tensor([t_val])

        sigma_ep = epdd.compute_epdd_noise_coefficient(x0, t)
        sigma_eff = epdd.compute_effective_noise_coefficient(x0, t)

        x_t = epdd.q_sample(x0, noise, t, C_vec, sigma_ep=sigma_ep)

        t_plus = torch.tensor([t_val + dt])
        sigma_ep_plus = epdd.compute_epdd_noise_coefficient(x0, t_plus)
        x_t_plus = epdd.q_sample(x0, noise, t_plus, C_vec, sigma_ep=sigma_ep_plus)

        numerical_deriv = (x_t_plus - x_t) / dt

        doc_velocity = C_vec + sigma_ep * noise
        strict_velocity = C_vec + sigma_eff * noise

        mae_doc = (doc_velocity - numerical_deriv).abs().mean().item()
        mae_strict = (strict_velocity - numerical_deriv).abs().mean().item()

        improvement = (mae_doc - mae_strict) / mae_doc * 100 if mae_doc > 0 else 0

        print(f"  {t_val:6.2f} | {mae_doc:14.6e} | {mae_strict:14.6e} | {improvement:10.2f}%")

    print()
    print("=" * 100)
    print("  验证完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
