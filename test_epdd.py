"""
EPDD (EdgeLatentDiffusion) 数值验证脚本

验证目标:
1. q_sample / pred_x0_from_xt 前向/反向一致性
2. 退化情况 (sigma_ep ≡ 1) 与 LatentDiffusion 等价
3. reverse_q_sample_c_list_concat 提取的 C 列表可用
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/root/autodl-tmp/CycleDiff')

from ddm.ddm_const_ode_2_learn import LatentDiffusion, EdgeLatentDiffusion
from ddm.encoder_decoder import AutoencoderKL


class DummyUnet(nn.Module):
    """用于测试的 Mock UNet，输出可学习的常数"""
    def __init__(self, channels=4):
        super().__init__()
        self.channels = channels
        self.self_condition = False
        # 输出 C 和 noise，初始化为 0
        self.C_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.noise_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, t, cond=None):
        # 返回与 x 同形状的 C 和 noise
        B, C, H, W = x.shape
        C_pred = self.C_bias.expand(B, C, H, W)
        noise_pred = self.noise_bias.expand(B, C, H, W)
        return C_pred, noise_pred


def create_models(image_size=(64, 64), z_channels=4):
    """创建测试所需的模型实例"""
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

    # LatentDiffusion (基线)
    ldm = LatentDiffusion(
        auto_encoder=auto_encoder,
        model=unet,
        image_size=image_size,
        scale_factor=1.0,
        scale_by_std=False,
        cfg=cfg,
        sampling_timesteps=10
    )

    # EdgeLatentDiffusion (EPDD)
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

    return ldm, epdd


def test_q_sample_pred_x0_consistency():
    """
    验证 6.1: q_sample / pred_x0_from_xt 前向/反向一致性

    理论依据:
    - EPDD 前向: x_t = (1-t)*x_0 + t*sigma_ep*noise
    - EPDD 反向: x_0 = (x_t - t*sigma_ep*noise) / (1-t)

    验证方法:
    1. 随机生成 x0, noise, t
    2. 用 q_sample 加噪得到 x_t
    3. 用 pred_x0_from_xt 从 x_t 恢复 x0_pred
    4. 验证 x0_pred ≈ x0 (误差 < 1e-4)
    """
    print("\n" + "=" * 80)
    print("Test 6.1: q_sample / pred_x0_from_xt 一致性验证")
    print("=" * 80)

    _, epdd = create_models()
    epdd.eval()

    B, C, H, W = 2, 4, 16, 16
    x0 = torch.randn(B, C, H, W)
    noise = torch.randn_like(x0)
    t = torch.tensor([0.3, 0.7])
    C_vec = -x0

    # 测试 1: 带 sigma_ep 的 EPDD 模式
    sigma_ep = epdd.compute_epdd_noise_coefficient(x0, t)
    x_t = epdd.q_sample(x0, noise, t, C_vec, sigma_ep=sigma_ep)
    x0_pred = epdd.pred_x0_from_xt(x_t, noise, C_vec, t, sigma_ep=sigma_ep)

    error = (x0_pred - x0).abs().max().item()
    print(f"  [EPDD mode] max reconstruction error: {error:.6e}")
    assert error < 1e-4, f"EPDD 模式重构误差过大: {error}"
    print("  ✓ EPDD 模式通过")

    # 测试 2: sigma_ep=None 的退化模式（应与基类一致）
    x_t_base = epdd.q_sample(x0, noise, t, C_vec, sigma_ep=None)
    x0_pred_base = epdd.pred_x0_from_xt(x_t_base, noise, C_vec, t, sigma_ep=None)

    error_base = (x0_pred_base - x0).abs().max().item()
    print(f"  [Base mode] max reconstruction error: {error_base:.6e}")
    assert error_base < 1e-4, f"退化模式重构误差过大: {error_base}"
    print("  ✓ 退化模式通过")

    # 测试 3: 边界情况 t->0
    t_small = torch.tensor([1e-3, 1e-3])
    sigma_ep_small = epdd.compute_epdd_noise_coefficient(x0, t_small)
    x_t_small = epdd.q_sample(x0, noise, t_small, C_vec, sigma_ep=sigma_ep_small)
    x0_pred_small = epdd.pred_x0_from_xt(x_t_small, noise, C_vec, t_small, sigma_ep=sigma_ep_small)

    error_small = (x0_pred_small - x0).abs().max().item()
    print(f"  [t->0] max reconstruction error: {error_small:.6e}")
    assert error_small < 1e-3, f"t->0 重构误差过大: {error_small}"
    print("  ✓ 边界情况 t->0 通过")

    print("  [Test 6.1 PASSED]")


def test_degradation_equivalence():
    """
    验证 6.2: 退化情况 (sigma_ep ≡ 1) 与 LatentDiffusion 等价

    理论依据:
    当 sigma_ep ≡ 1 时，EPDD 的前向/反向公式退化为标准解耦扩散:
    - x_t = (1-t)*x_0 + t*noise = x_0 + C*t + t*noise (因为 C=-x0)
    - x_0 = x_t - C*t - t*noise

    这与 LatentDiffusion (继承自 DDPM) 的 q_sample / pred_x0_from_xt 一致。

    验证方法:
    1. 对同一组 (x0, noise, t, C)，比较 EPDD(sigma_ep=1) 和 LatentDiffusion 的输出
    2. 比较 q_sample 输出
    3. 比较 pred_x0_from_xt 输出
    """
    print("\n" + "=" * 80)
    print("Test 6.2: 退化情况 (sigma_ep ≡ 1) 与 LatentDiffusion 等价性")
    print("=" * 80)

    ldm, epdd = create_models()
    ldm.eval()
    epdd.eval()

    B, C, H, W = 2, 4, 16, 16
    x0 = torch.randn(B, C, H, W)
    noise = torch.randn_like(x0)
    t = torch.tensor([0.3, 0.7])
    C_vec = -x0

    # 创建 sigma_ep ≡ 1 的张量
    sigma_ep_ones = torch.ones(B, 1, H, W)

    # 比较 q_sample
    x_t_ldm = ldm.q_sample(x0, noise, t, C_vec)
    x_t_epdd = epdd.q_sample(x0, noise, t, C_vec, sigma_ep=sigma_ep_ones)

    q_diff = (x_t_ldm - x_t_epdd).abs().max().item()
    print(f"  q_sample diff (LatentDiffusion vs EPDD with sigma_ep=1): {q_diff:.6e}")
    assert q_diff < 1e-5, f"q_sample 不匹配: {q_diff}"
    print("  ✓ q_sample 等价性通过")

    # 比较 pred_x0_from_xt
    x0_pred_ldm = ldm.pred_x0_from_xt(x_t_ldm, noise, C_vec, t)
    x0_pred_epdd = epdd.pred_x0_from_xt(x_t_epdd, noise, C_vec, t, sigma_ep=sigma_ep_ones)

    pred_diff = (x0_pred_ldm - x0_pred_epdd).abs().max().item()
    print(f"  pred_x0_from_xt diff: {pred_diff:.6e}")
    assert pred_diff < 1e-5, f"pred_x0_from_xt 不匹配: {pred_diff}"
    print("  ✓ pred_x0_from_xt 等价性通过")

    # 额外验证：sigma_ep=None 时也应等价
    x_t_epdd_none = epdd.q_sample(x0, noise, t, C_vec, sigma_ep=None)
    x0_pred_epdd_none = epdd.pred_x0_from_xt(x_t_epdd_none, noise, C_vec, t, sigma_ep=None)

    q_diff_none = (x_t_ldm - x_t_epdd_none).abs().max().item()
    pred_diff_none = (x0_pred_ldm - x0_pred_epdd_none).abs().max().item()
    print(f"  q_sample diff (sigma_ep=None): {q_diff_none:.6e}")
    print(f"  pred_x0_from_xt diff (sigma_ep=None): {pred_diff_none:.6e}")
    assert q_diff_none < 1e-5, f"sigma_ep=None 时 q_sample 不匹配"
    assert pred_diff_none < 1e-5, f"sigma_ep=None 时 pred_x0_from_xt 不匹配"
    print("  ✓ sigma_ep=None 退化模式等价性通过")

    print("  [Test 6.2 PASSED]")


def test_reverse_q_sample_c_list():
    """
    验证 6.3: reverse_q_sample_c_list_concat 提取的 C 列表可用

    验证方法:
    1. 构造一个简单输入 x_start
    2. 调用 reverse_q_sample_c_list_concat 提取 c_list
    3. 验证 c_list 非空且每个元素形状正确
    4. 验证 c_list 的最后一个元素是 t=0 时刻的 C (理论上应 ≈ -x0)
    5. 使用 sample_fn_d_c_list 基于 c_list 生成图像，验证流程不报错
    """
    print("\n" + "=" * 80)
    print("Test 6.3: reverse_q_sample_c_list_concat C 列表可用性")
    print("=" * 80)

    _, epdd = create_models(image_size=(64, 64))
    epdd.eval()

    # 构造测试输入 (模拟 VAE 编码后的潜变量)
    B, C, H, W = 1, 4, 16, 16
    x_start = torch.randn(B, 3, 64, 64)

    # 为了测试 reverse_q_sample_c_list_concat，我们需要模拟 VAE 输出
    # 由于 VAE 需要训练，我们直接构造一个 mock 的 first_stage_model
    class MockVAE(nn.Module):
        def __init__(self, z_channels=4, down_ratio=4):
            super().__init__()
            self.z_channels = z_channels
            self.down_ratio = down_ratio
            self.encode = lambda x: torch.randn(B, z_channels, H, W)
            self.decode = lambda z: torch.randn(B, 3, 64, 64)

    mock_vae = MockVAE()
    epdd.first_stage_model = mock_vae

    # 替换 model 为返回确定值的 mock
    class DeterministicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.channels = C
            self.self_condition = False
        def forward(self, x, t, cond=None):
            # 返回理想的 C 和 noise
            # C = -x0, noise = 0 (理想情况)
            return -x, torch.zeros_like(x)

    epdd.model = DeterministicModel()

    print("  Running reverse_q_sample_c_list_concat...")
    c_list, x_t = epdd.reverse_q_sample_c_list_concat(x_start)

    # 验证 c_list 结构
    assert len(c_list) == epdd.sampling_timesteps, \
        f"c_list 长度应为 {epdd.sampling_timesteps}, 实际为 {len(c_list)}"
    print(f"  ✓ c_list 长度正确: {len(c_list)}")

    # 验证每个 C 的形状
    for i, c in enumerate(c_list):
        assert c.shape == (B, C, H, W), f"c_list[{i}] 形状错误: {c.shape}"
    print(f"  ✓ 所有 C 元素形状正确: {(B, C, H, W)}")

    # 验证 c_list 的最后一个元素 (t=0 时刻的 C)
    # 在理想情况下，C = -x0，其中 x0 是初始潜变量
    last_C = c_list[-1]
    print(f"  Last C stats: mean={last_C.mean():.4f}, std={last_C.std():.4f}")

    # 验证 sample_fn_d_c_list 可用
    print("  Testing sample_fn_d_c_list with extracted c_list...")
    c_list_copy = [c.clone() for c in c_list]  # 复制一份，因为 pop 会修改列表
    try:
        generated = epdd.sample_fn_d_c_list(
            (B, C, H, W),
            c_list=c_list_copy,
            unnormalize=False
        )
        assert generated.shape == (B, C, H, W), f"生成图像形状错误: {generated.shape}"
        print(f"  ✓ sample_fn_d_c_list 运行成功，输出形状: {generated.shape}")
    except Exception as e:
        print(f"  ✗ sample_fn_d_c_list 运行失败: {e}")
        raise

    # 验证 sample_from_c_list 可用
    print("  Testing sample_from_c_list...")
    c_list_copy2 = [c.clone() for c in c_list]
    try:
        img_rec = epdd.sample_from_c_list(
            batch_size=B,
            c_list=c_list_copy2
        )
        assert img_rec.shape[0] == B, f"生成批次大小错误: {img_rec.shape[0]}"
        print(f"  ✓ sample_from_c_list 运行成功，输出形状: {img_rec.shape}")
    except Exception as e:
        print(f"  ✗ sample_from_c_list 运行失败: {e}")
        raise

    print("  [Test 6.3 PASSED]")


def test_edge_preservation_property():
    """
    额外验证: 边缘保持性质

    理论依据:
    - 在边缘区域 (||∇z0|| 大)，sigma_ep 应较小 (< 1)，减少噪声注入以保护边缘
    - 在平坦区域 (||∇z0|| 小)，sigma_ep 应接近 1

    验证方法:
    1. 构造一个具有明显边缘的潜变量 (如阶跃函数)
    2. 计算 sigma_ep
    3. 验证边缘处的 sigma_ep < 平坦处的 sigma_ep
    """
    print("\n" + "=" * 80)
    print("Extra Test: 边缘保持性质验证")
    print("=" * 80)

    _, epdd = create_models()
    epdd.eval()

    # 构造阶跃边缘图像 [1, 1, 32, 32]
    z = torch.zeros(1, 1, 32, 32)
    z[:, :, :, 16:] = 1.0  # 右半边为 1，左半边为 0，形成垂直边缘

    # 扩展到 4 通道
    z = z.repeat(1, 4, 1, 1)

    t = torch.tensor([0.3])
    sigma_ep = epdd.compute_epdd_noise_coefficient(z, t)

    # 提取边缘区域和平坦区域的 sigma_ep
    # 边缘在 x=16 附近
    edge_region = sigma_ep[:, :, :, 14:18]  # 边缘附近
    flat_region_left = sigma_ep[:, :, :, 2:6]   # 左平坦区域
    flat_region_right = sigma_ep[:, :, :, 26:30]  # 右平坦区域

    edge_mean = edge_region.mean().item()
    flat_left_mean = flat_region_left.mean().item()
    flat_right_mean = flat_region_right.mean().item()

    print(f"  Edge region sigma_ep: {edge_mean:.4f}")
    print(f"  Flat left sigma_ep: {flat_left_mean:.4f}")
    print(f"  Flat right sigma_ep: {flat_right_mean:.4f}")

    # 边缘处的 sigma_ep 应小于平坦处
    assert edge_mean < flat_left_mean, \
        f"边缘处 sigma_ep ({edge_mean}) 应小于平坦处 ({flat_left_mean})"
    assert edge_mean < flat_right_mean, \
        f"边缘处 sigma_ep ({edge_mean}) 应小于平坦处 ({flat_right_mean})"
    print("  ✓ 边缘保持性质验证通过 (边缘处 sigma_ep < 平坦处)")

    print("  [Extra Test PASSED]")


def test_training_step():
    """
    额外验证: 训练步骤能否正常运行
    """
    print("\n" + "=" * 80)
    print("Extra Test: 训练步骤运行验证")
    print("=" * 80)

    _, epdd = create_models(image_size=(64, 64))
    epdd.train()

    # Mock VAE
    class MockVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_ratio = 4
        def encode(self, x):
            return torch.randn(x.shape[0], 4, 16, 16)
        def decode(self, z):
            return torch.randn(z.shape[0], 3, 64, 64)

    epdd.first_stage_model = MockVAE()

    image = torch.rand(2, 3, 64, 64)
    mask = torch.rand(2, 1, 64, 64)
    batch = {'image': image, 'cond': mask}

    try:
        loss, loss_dict = epdd.training_step(batch)
        print(f"  ✓ Training step successful")
        print(f"    Loss: {loss.item():.4f}")
        print(f"    Loss simple: {loss_dict['train/loss_simple'].item():.6f}")
        print(f"    Loss vlb: {loss_dict['train/loss_vlb'].item():.6f}")
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        raise

    print("  [Extra Test PASSED]")


def run_all_tests():
    """运行所有验证测试"""
    print("\n" + "#" * 80)
    print("# EPDD (EdgeLatentDiffusion) 数值验证套件")
    print("#" * 80)

    torch.manual_seed(42)

    try:
        test_q_sample_pred_x0_consistency()
        test_degradation_equivalence()
        test_reverse_q_sample_c_list()
        test_edge_preservation_property()
        test_training_step()

        print("\n" + "#" * 80)
        print("# 所有测试通过! ✓")
        print("#" * 80)
        return True

    except AssertionError as e:
        print(f"\n  ✗ 验证失败: {e}")
        return False
    except Exception as e:
        print(f"\n  ✗ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
