import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ddm.ema import EMA
from ddm.utils import *
import torchvision as tv
from ddm.encoder_decoder import AutoencoderKL
from ddm.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
import swanlab
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument(
        "--cfg", help="experiment configure file name", type=str, required=True
    )
    args = parser.parse_args()
    args.cfg_path = args.cfg
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


def extract_experiment_name(cfg_path):
    """从配置文件路径提取实验名称"""
    if not cfg_path:
        return "ldm_experiment"

    match = re.search(r"([^/]+)\.yaml$", cfg_path)
    if match:
        return match.group(1)

    basename = os.path.basename(cfg_path)
    if basename:
        return basename.replace(".yaml", "").replace(".yml", "")

    return "ldm_experiment"


def main(args):
    cfg = CfgNode(args.cfg)
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = construct_class_by_name(**first_stage_cfg)
    unet_cfg = model_cfg.unet
    unet = construct_class_by_name(**unet_cfg)
    model_kwargs = {"model": unet,
                    "auto_encoder": first_stage_model, "cfg": model_cfg}
    model_kwargs.update(model_cfg)
    ldm = construct_class_by_name(**model_kwargs)
    model_kwargs.pop("model")
    model_kwargs.pop("auto_encoder")

    data_cfg = cfg.data
    dataset = construct_class_by_name(**data_cfg)
    dl = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=data_cfg.get("num_workers", 2),
    )
    train_cfg = cfg.trainer

    experiment_name = extract_experiment_name(args.cfg_path)

    swanlab.init(
        project="CycleDiff-RSI2Map",
        experiment_name=experiment_name,
        description=f"LDM training for RSI2Map task using {experiment_name} configuration",
        config={
            "model_class": model_cfg.get("class_name", "LatentDiffusion"),
            "batch_size": data_cfg.get("batch_size", 16),
            "learning_rate": train_cfg.get("lr", 1e-4),
            "min_lr": train_cfg.get("min_lr", 1e-6),
            "train_num_steps": train_cfg.get("train_num_steps", 100000),
            "image_size": model_cfg.get("image_size", [256, 256]),
            "gradient_accumulate_every": train_cfg.get("gradient_accumulate_every", 1),
            "save_and_sample_every": train_cfg.get("save_and_sample_every", 1000),
            "amp": train_cfg.get("amp", False),
            "fp16": train_cfg.get("fp16", False),
            "sampling_timesteps": model_cfg.get("sampling_timesteps", 100),
        },
    )

    trainer = Trainer(
        ldm,
        dl,
        train_batch_size=data_cfg.batch_size,
        gradient_accumulate_every=train_cfg.gradient_accumulate_every,
        train_lr=train_cfg.lr,
        train_num_steps=train_cfg.train_num_steps,
        save_and_sample_every=train_cfg.save_and_sample_every,
        results_folder=train_cfg.results_folder,
        amp=train_cfg.amp,
        fp16=train_cfg.fp16,
        log_freq=train_cfg.log_freq,
        cfg=cfg,
        resume_milestone=train_cfg.resume_milestone,
        train_wd=train_cfg.get("weight_decay", 1e-4),
    )

    if train_cfg.test_before:
        with torch.no_grad():
            for datatmp in dl:
                break
            all_images = trainer.model.sample(batch_size=data_cfg.batch_size)

        nrow = 2 ** math.floor(math.log2(math.sqrt(data_cfg.batch_size)))
        tv.utils.save_image(
            all_images,
            str(
                trainer.results_folder
                / f"sample-{train_cfg.resume_milestone}_{model_cfg.sampling_timesteps}.png"
            ),
            nrow=nrow,
        )
        torch.cuda.empty_cache()

    trainer.train()
    swanlab.finish()
    pass


class Trainer(object):
    def __init__(
        self,
        model,
        data_loader,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_wd=1e-4,
        train_num_steps=100000,
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        fp16=False,
        log_freq=20,
        resume_milestone=0,
        cfg={},
    ):
        super().__init__()

        # 设置设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 混合精度配置
        self.amp = amp
        self.fp16 = fp16
        if fp16:
            self.dtype = torch.float16
        elif amp:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # 梯度缩放器（用于混合精度训练）
        self.scaler = torch.amp.GradScaler('cuda') if (
            amp or fp16) and self.device.type == 'cuda' else None

        self.cfg = cfg

        self.model = model.to(self.device)

        assert has_int_squareroot(
            num_samples
        ), "number of samples must have an integer square root"

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq

        self.train_num_steps = train_num_steps
        self.image_size = model.image_size

        dl = data_loader
        self.dl = cycle(dl)

        def WarmUpLrScheduler(iter):
            warmup_iter = cfg.trainer.get("warmup_iter", 5000)
            if iter <= warmup_iter:
                ratio = (iter + 1) / warmup_iter
            else:
                ratio = max(
                    (1 - (iter - warmup_iter) / train_num_steps) ** 0.96,
                    cfg.trainer.min_lr / train_lr,
                )
            return ratio

        self.opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_lr,
            weight_decay=train_wd,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=WarmUpLrScheduler
        )

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.ema = EMA(
            model,
            ema_model=None,
            beta=0.9996,
            update_after_step=cfg.trainer.ema_update_after_step,
            update_every=cfg.trainer.ema_update_every,
        )
        self.ema.to(self.device)

        self.step = 0

        resume_file = str(self.results_folder / f"model-{resume_milestone}.pt")
        if os.path.isfile(resume_file):
            self.load(resume_milestone)

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        device = self.device

        data = safe_torch_load(
            str(self.results_folder / f"model-{milestone}.pt"),
            map_location=device,
        )

        self.model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])
        self.ema.load_state_dict(data["ema"])
        if "scale_factor" in data["model"]:
            # 恢复训练时始终使用检查点中保存的 scale_factor
            # 以保证训练的一致性和连续性
            self.model.scale_factor = data["model"]["scale_factor"]

        if self.scaler and data.get("scaler"):
            self.scaler.load_state_dict(data["scaler"])

        print(f"### USING DEFAULT SCALE {self.model.scale_factor}")

    def train(self):
        device = self.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
        ) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.0
                total_loss_dict = {
                    "loss_simple": 0.0,
                    "loss_vlb": 0.0,
                    "total_loss": 0.0,
                    "lr": 5e-5,
                }
                for ga_ind in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device)
                if isinstance(self.model, nn.Module):
                    self.model.on_train_batch_start(batch)

                    # 混合精度训练
                    with torch.cuda.amp.autocast(enabled=self.amp or self.fp16, dtype=self.dtype):
                        loss, log_dict = self.model.training_step(batch)

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        loss_simple = (
                            log_dict["train/loss_simple"].item()
                            / self.gradient_accumulate_every
                        )
                        loss_vlb = (
                            log_dict["train/loss_vlb"].item()
                            / self.gradient_accumulate_every
                        )
                        total_loss_dict["loss_simple"] += loss_simple
                        total_loss_dict["loss_vlb"] += loss_vlb

                    # 反向传播
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                total_loss_dict["total_loss"] = total_loss
                total_loss_dict["lr"] = self.opt.param_groups[0]["lr"]
                describtions = dict2str(total_loss_dict)
                describtions = (
                    "[Train Step] {}/{}: ".format(self.step,
                                                  self.train_num_steps)
                    + describtions
                )
                pbar.desc = describtions

                if self.step % self.log_freq == 0:
                    print(describtions)

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()), 1.0)

                # 优化器步骤
                if self.scaler:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                self.opt.zero_grad()
                self.lr_scheduler.step()
                swanlab.log(
                    {
                        "Learning_Rate": total_loss_dict["lr"],
                        "total_loss": total_loss_dict["total_loss"],
                        "loss_simple": total_loss_dict["loss_simple"],
                        "loss_vlb": total_loss_dict["loss_vlb"],
                    }
                )

                self.step += 1
                self.ema.to(device)
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)
                    self.model.eval()

                    all_images = self.model.sample(batch_size=16)

                    nrow = 2 ** math.floor(math.log2(math.sqrt(16)))
                    sample_path = str(
                        self.results_folder / f"sample-{milestone}.png"
                    )
                    tv.utils.save_image(all_images, sample_path, nrow=nrow)

                    try:
                        swanlab.log(
                            {
                                "generated_images": swanlab.Image(
                                    sample_path,
                                    caption=f"Step {self.step} - Milestone {milestone}",
                                )
                            }
                        )
                    except Exception as e:
                        print(f"Failed to log images to SwanLab: {e}")

                    self.model.train()

                pbar.update(1)

        print("training complete")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass
