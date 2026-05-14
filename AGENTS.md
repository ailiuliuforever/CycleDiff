# AGENTS.md — CycleDiff

## Setup

- Python 3.9, PyTorch 1.13.1+cu117. Install: `pip install -r requirement.txt`
- The vendored `src/torch-fidelity/` is a dependency; do not modify it
- No CI, no pre-commit, no lint/formatter — only `pyrightconfig.json` for type checking (covers `ddm/`, `unet/`, `util/`, `torch_utils/`, `metrics/`)

## Architecture

- `ddm/` — diffusion model core (DDPM, LatentDiffusion, data loading, cycle translator trainers, EMA, augment)
- `unet/` — UNet backbones (DhariwalUNet, EDMPrecond, cond/uncond variants)
- `taming/` — VAE autoencoder modules and perceptual losses
- `configs/` — one subdirectory per dataset pair (e.g. `afhq_cat2dog/`, `rsi2map_AIDOMG_Beijing/`)
- `util/` — evaluation metrics (FID, PSNR, SSIM, gradient loss)
- `metrics/` — quantitative metrics pipeline
- `torch_utils/` — distributed training helpers

Config-driven: YAML configs use `class_name` keys resolved by `ddm.utils.construct_class_by_name`. All paths in configs (data_root, ckpt_path, results_folder) are hardcoded absolute paths — must be updated per machine.

## Training pipeline (strict order)

1. **VAE**: `accelerate launch train_vae.py --cfg ./configs/{dataset}/{class}_ae_kl_256x256_d4.yaml`
2. **LDM**: `accelerate launch train_uncond_ldm.py --cfg ./configs/{dataset}/{class}_ddm_const4_ldm_unet6_114_ode_2.yaml`
3. **Cycle Translator** (single GPU): `python train_uncond_ldm_cycle_swanlab.py --cfg ./configs/{dataset}/translation_C_disc_timestep_ode_2.yaml`
   - Multi-GPU variant: `accelerate launch train_uncond_ldm_cycle_multi_gpu.py --cfg ...`
   - `train_uncond_ldm_cycle.py` is a legacy script — no gradient loss, prefer the swanlab version

Each stage depends on the previous. LDM needs VAE checkpoint; Cycle Translator needs both LDM checkpoints (`ckpt_path1`, `ckpt_path2` in config).

## Testing

Single test file: `python test_epdd.py` — validates EPDD (EdgeLatentDiffusion) numerical correctness. No other automated tests.

## Translation inference

`accelerate launch translation_uncond_ldm_cycle.py --cfg ./configs/{dataset}/translation_C_disc_timestep_ode_2.yaml`

## Experiment scripts

- `run_experiments.sh` — runs paired cgrad+b baseline experiments sequentially using SwanLab scripts. Intended for short validation runs.
- Experiment configs (maps dataset):
  - `translation_C_disc_timestep_ode_2.yaml` — full training (200k steps, `c_gradient_weight: 10.0`)
  - `translation_cgrad.yaml` — gradient loss validation (3k steps, `c_gradient_weight: 3.0`)
  - `translation_baseline.yaml` — baseline w/o gradient loss (3k steps, `c_gradient_weight: 0.0`)

## Evaluation

- `evaluation/cyclediff/evaluate_cyclediff.py` — comprehensive eval: A2B translation, cycle consistency (ABA), identity mapping, C-space per-timestep visualization. 1857 lines, no separate CLI — modify paths and run directly.
- `verify_sigma_eff.py` — diffusion sigma efficiency debugging utility (not a test).
- Pre-computed results:
  - `evaluation/cyclediff/res/` — baseline model results (includes `evaluation_metrics.txt`)
  - `evaluation/cyclediff/res_cgrad/` — gradient loss experiment results

## Key gotchas

- **Pretrained VAE weight**: Download from `https://ommer-lab.com/files/latent-diffusion/kl-f4.zip`, place in `kl-f4/`, set `ckpt_path` in `*_ae_kl_256x256_d4.yaml` (line 19)
- **Dataset structure**: `{root}/{split}/{class_A}/`, `{root}/{split}/{class_B}/` (see README)
- **Gradient accumulation**: Cycle translator uses `gradient_accumulate_every=2` with GA step 0 for generator loss and GA step 1 for discriminator loss — don't change this without understanding the alternating update pattern
- **`safe_torch_load`** (in `ddm/utils.py:17`): Use this instead of raw `torch.load` — handles PyTorch 2.6+ `weights_only` default change
- **EMA**: All models use EMA with configurable `ema_update_after_step` and `ema_update_every`. Checkpoints may store EMA weights prefixed with `ema_model.` — the `ft_use_ema` flag in config controls loading behavior
- **Config paths are absolute**: Every YAML config has hardcoded paths like `/data1/zoushilong/...`. Modify `data_root`, `results_folder`, `ckpt_path*`, and `save_folder` for your machine
- **`scale_factor`**: Stored in checkpoints, must match config; the LDM loader checks for mismatch and can override
- **`src/` directory** is gitignored but contains vendored `torch-fidelity` and is required at runtime
- **C-gradient loss**: Built into `train_uncond_ldm_cycle_swanlab.py`. Config keys `c_gradient_weight` and `c_gradient_edge_boost` control it. The legacy scripts `train_uncond_ldm_cycle.py` and `train_uncond_ldm_cycle_multi_gpu.py` do not implement gradient loss — these parameters are silently ignored if using those scripts.
- **SwanLab logging**: Training scripts named `*_swanlab.py` log to SwanLab instead of TensorBoard. `train_uncond_ldm_cycle_swanlab.py` is the primary cycle translator training script.
