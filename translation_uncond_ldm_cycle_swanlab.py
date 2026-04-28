import numpy as np
import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from ddm.utils import *
import torchvision as tv
from ddm.encoder_decoder import AutoencoderKL
from ddm.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
from scipy import integrate
from util.eval_score import fid_l2_psnr_ssim
import swanlab
from datetime import datetime
import time


def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


def cfgnode_to_dict(cfg_node):
    if isinstance(cfg_node, CfgNode):
        return {k: cfgnode_to_dict(v) for k, v in cfg_node.items()}
    elif isinstance(cfg_node, dict):
        return {k: cfgnode_to_dict(v) for k, v in cfg_node.items()}
    else:
        return cfg_node


def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)

    model_cfg1 = cfg.model1
    assert model_cfg1.ldm, 'This file is only used for ldm！'
    first_stage_cfg1 = model_cfg1.first_stage
    first_stage_model1 = construct_class_by_name(**first_stage_cfg1)
    unet_cfg1 = model_cfg1.unet
    unet1 = construct_class_by_name(**unet_cfg1)
    model_kwargs1 = {'model': unet1, 'auto_encoder': first_stage_model1, 'cfg': model_cfg1}
    model_kwargs1.update(model_cfg1)
    ldm1 = construct_class_by_name(**model_kwargs1)
    model_kwargs1.pop('model')
    model_kwargs1.pop('auto_encoder')

    model_cfg2 = cfg.model2
    first_stage_cfg2 = model_cfg2.first_stage
    first_stage_model2 = construct_class_by_name(**first_stage_cfg2)
    unet_cfg2 = model_cfg2.unet
    unet2 = construct_class_by_name(**unet_cfg2)
    model_kwargs2 = {'model': unet2, 'auto_encoder': first_stage_model2, 'cfg': model_cfg2}
    model_kwargs2.update(model_cfg2)
    ldm2 = construct_class_by_name(**model_kwargs2)
    model_kwargs2.pop('model')
    model_kwargs2.pop('auto_encoder')

    net_G_A_cfg = cfg.net_G
    net_G_A = construct_class_by_name(**net_G_A_cfg)
    net_G_B_cfg = cfg.net_G
    net_G_B = construct_class_by_name(**net_G_B_cfg)

    if cfg.sampler.task == "cat2dog":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "dog2cat":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "wild2dog":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "dog2wild":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "male2female":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "female2male":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "sem2rgb":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "rgb2sem":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "edge2rgb":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "rgb2edge":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "depth2rgb":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "rgb2depth":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "summer2winter":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "winter2summer":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "horse2zebra":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "zebra2horse":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "young2old":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "old2young":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "map2satellite":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "satellite2map":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "label2cityscape":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "cityscape2label":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "rsi2map":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "map2rsi":
        data_cfg = cfg.data_test2
    else:
        raise ValueError(f"Unknown task: {cfg.sampler.task}. Supported tasks include: cat2dog, dog2cat, wild2dog, dog2wild, male2female, female2male, sem2rgb, rgb2sem, edge2rgb, rgb2edge, depth2rgb, rgb2depth, summer2winter, winter2summer, horse2zebra, zebra2horse, young2old, old2young, map2satellite, satellite2map, label2cityscape, cityscape2label, rsi2map, map2rsi")

    dataset = construct_class_by_name(**data_cfg)
    dl = DataLoader(dataset, batch_size=cfg.sampler.batch_size, shuffle=False, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))

    sampler_cfg = cfg.sampler
    sampler = Sampler(
        ldm1, ldm2, net_G_A, net_G_B, dl, batch_size=sampler_cfg.batch_size,
        results_folder=sampler_cfg.save_folder,cfg=cfg,
    )
    sampler.sample()
    if sampler_cfg.get('cal_metrics', False):
        sampler.cal_metrics(task=sampler_cfg.task, source_gt_path=sampler_cfg.source_gt_path, target_gt_path=sampler_cfg.target_gt_path)
    
    if sampler.accelerator.is_main_process:
        swanlab.finish()
    pass


class Sampler(object):
    def __init__(
            self,
            model1,
            model2,
            net_G_A,
            net_G_B,
            data_loader,
            batch_size=16,
            results_folder='./results',
            rk45=False,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='no',
            kwargs_handlers=[ddp_handler],
        )
        self.model1 = model1
        self.model2 = model2
        self.net_G_A = net_G_A
        self.net_G_B = net_G_B
        self.rk45 = rk45

        self.batch_size = batch_size

        self.image_size = model1.image_size

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.cfg = cfg
        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)

        self.model1, self.model2, self.net_G_A, self.net_G_B = self.accelerator.prepare(self.model1, self.model2, self.net_G_A, self.net_G_B)
        data = safe_torch_load(cfg.sampler.ckpt_path,
                          map_location=lambda storage, loc: storage)

        self.model1 = self.accelerator.unwrap_model(self.model1)
        self.model2 = self.accelerator.unwrap_model(self.model2)
        self.net_G_A = self.accelerator.unwrap_model(self.net_G_A)
        self.net_G_B = self.accelerator.unwrap_model(self.net_G_B)

        if cfg.sampler.use_ema:
            sd_d1 = data['ema_d1']
            new_sd = {}
            for k in sd_d1.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]
                    new_sd[new_k] = sd_d1[k]
            sd_d1 = new_sd
            self.model1.load_state_dict(sd_d1)
            sd_d2 = data['ema_d2']
            new_sd = {}
            for k in sd_d2.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]
                    new_sd[new_k] = sd_d2[k]
            sd_d2 = new_sd
            self.model2.load_state_dict(sd_d2)

            sd_G_A = data['ema_G_A']
            new_sd = {}
            for k in sd_G_A.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]
                    new_sd[new_k] = sd_G_A[k]
            sd_G_A = new_sd
            self.net_G_A.load_state_dict(sd_G_A)

            sd_G_B = data['ema_G_B']
            new_sd = {}
            for k in sd_G_B.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]
                    new_sd[new_k] = sd_G_B[k]
            sd_G_B = new_sd
            self.net_G_B.load_state_dict(sd_G_B)
        else:
            self.model1.load_state_dict(data['model1'])
            self.model2.load_state_dict(data['model2'])
            self.net_G_A.load_state_dict(data['net_G_A'])
            self.net_G_B.load_state_dict(data['net_G_B'])
        if 'scale_factor' in data['model1']:
            self.model1.scale_factor = data['model1']['scale_factor']
            self.model2.scale_factor = data['model2']['scale_factor']

        if self.accelerator.is_main_process:
            config_dict = cfgnode_to_dict(cfg)
            task_name = cfg.sampler.task
            swanlab.init(
                project="CycleDiff-ImageTranslation",
                experiment_name=f"translation_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config_dict,
                description=f"Image-to-Image Translation: {task_name}"
            )

    def get_latent_space(self, x, tag=None):
        if "src" in tag:
            z = self.model1.first_stage_model.encode(x)
            z = self.model1.get_first_stage_encoding(z)
            z = self.model1.scale_factor * z
        elif "trg" in tag:
            z = self.model2.first_stage_model.encode(x)
            z = self.model2.get_first_stage_encoding(z)
            z = self.model2.scale_factor * z
        return z

    def sample(self):
        accelerator = self.accelerator
        device = accelerator.device
        task = self.cfg.sampler.task
        log_image_freq = self.cfg.sampler.get('log_image_freq', 10)
        
        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()
            self.net_G_A.eval()
            self.net_G_B.eval()

            start_time = time.time()
            
            for idx, batch in tqdm(enumerate(self.dl), total=len(self.dl)):
                batch_start_time = time.time()
                
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key].to(device)

                if "cat2dog" in self.cfg.sampler.task or "wild2dog" in self.cfg.sampler.task or "male2female" in self.cfg.sampler.task or "sem2rgb" in self.cfg.sampler.task or\
                    "depth2rgb" in self.cfg.sampler.task or "edge2rgb" in self.cfg.sampler.task or "summer2winter" in self.cfg.sampler.task or "horse2zebra" in self.cfg.sampler.task or \
                    "young2old" in self.cfg.sampler.task or "map2satellite" in self.cfg.sampler.task or "label2cityscape" in self.cfg.sampler.task or "rsi2map" in self.cfg.sampler.task:
                    src_img = batch["image"]
                    x_s = self.get_latent_space(src_img, tag="src_img")

                    c_list, noise = self.model1.reverse_q_sample_c_list_concat(src_img)
                    target_input = []

                    step = 1. / self.model1.sampling_timesteps
                    rho = 1.
                    step_indices = torch.arange(self.model1.sampling_timesteps, dtype=torch.float32, device=device)
                    t_steps = (self.model1.sigma_max ** (1 / rho) + step_indices / (self.model1.sampling_timesteps - 1) * (
                            step - self.model1.sigma_max ** (1 / rho))) ** rho
                    t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))

                    for i in range(len(c_list[:-1])):
                        target_input.append(self.net_G_A(c_list[i], t_steps[i+1].repeat((x_s.shape[0],))))

                    target_input.append(self.net_G_A(c_list[-1], t_steps[-1].repeat((x_s.shape[0],))))
                    target_input.append(noise)
                    pred_img = self.model2.sample_from_c_list(batch_size=src_img.shape[0], c_list=target_input)

                else:
                    trg_img = batch["image"]
                    x_t = self.get_latent_space(trg_img, tag="trg_img")

                    c_list2, noise2 = self.model2.reverse_q_sample_c_list_concat(trg_img)
                    target_input = []

                    step = 1. / self.model2.sampling_timesteps
                    rho = 1.
                    step_indices = torch.arange(self.model2.sampling_timesteps, dtype=torch.float32, device=device)
                    t_steps = (self.model2.sigma_max ** (1 / rho) + step_indices / (self.model2.sampling_timesteps - 1) * (
                            step - self.model2.sigma_max ** (1 / rho))) ** rho
                    t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))

                    for i in range(len(c_list2[:-1])):
                        target_input.append(self.net_G_B(c_list2[i], t_steps[i+1].repeat((x_t.shape[0],))))

                    target_input.append(self.net_G_B(c_list2[-1], t_steps[-1].repeat((x_t.shape[0],))))
                    target_input.append(noise2)
                    pred_img = self.model1.sample_from_c_list(batch_size=trg_img.shape[0], c_list=target_input)

                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time

                if accelerator.is_main_process:
                    total_time = time.time() - start_time
                    samples_processed = (idx + 1) * self.batch_size
                    
                    swanlab.log({
                        "progress/batch_index": idx,
                        "progress/samples_processed": samples_processed,
                        "progress/batch_time": batch_time,
                        "progress/total_time": total_time,
                    }, step=idx)

                    if idx % log_image_freq == 0:
                        if "cat2dog" in task or "wild2dog" in task or "male2female" in task or "sem2rgb" in task or \
                           "depth2rgb" in task or "edge2rgb" in task or "summer2winter" in task or "horse2zebra" in task or \
                           "young2old" in task or "map2satellite" in task or "label2cityscape" in task or "rsi2map" in task:
                            source_image = src_img
                        else:
                            source_image = trg_img
                        
                        try:
                            # 确保传入 swanlab.Image 的数据在 [0, 1] 范围
                            # swanlab.Image 内部使用 torchvision.utils.make_grid(..., normalize=True)
                            # 如果数据范围已经是 [0, 1]，需要避免被重新归一化
                            # 通过转换为 PIL.Image 来绕过 make_grid 的 normalize 处理
                            from torchvision.transforms import ToPILImage
                            to_pil = ToPILImage()
                            
                            # 处理 source_image: 可能是 [-1, 1] 或 [0, 1]
                            if isinstance(source_image, torch.Tensor):
                                if source_image.min() < 0:
                                    source_display = (source_image + 1) / 2
                                else:
                                    source_display = source_image
                                source_display = torch.clamp(source_display, 0, 1)
                                # 转换为 PIL.Image 避免 make_grid 的 normalize
                                if source_display.dim() == 4:
                                    source_pil = to_pil(source_display[0].cpu())
                                else:
                                    source_pil = to_pil(source_display.cpu())
                            else:
                                source_pil = source_image
                            
                            # 处理 pred_img: 应该已经是 [0, 1]，但确保无误
                            if isinstance(pred_img, torch.Tensor):
                                pred_display = torch.clamp(pred_img, 0, 1)
                                if pred_display.dim() == 4:
                                    pred_pil = to_pil(pred_display[0].cpu())
                                else:
                                    pred_pil = to_pil(pred_display.cpu())
                            else:
                                pred_pil = pred_img
                            
                            swanlab.log({
                                "translation_samples/source": swanlab.Image(
                                    source_pil,
                                    caption=f"Task: {task}, Batch: {idx}, Source Images"
                                ),
                                "translation_samples/translated": swanlab.Image(
                                    pred_pil,
                                    caption=f"Task: {task}, Batch: {idx}, Translated Images"
                                )
                            }, step=idx)
                        except Exception as e:
                            print(f"Warning: Failed to log images to SwanLab at batch {idx}: {e}")

                for j in range(pred_img.shape[0]):
                    img = pred_img[j]
                    file_name = batch["img_name"][j]
                    file_name = self.results_folder / file_name
                    tv.utils.save_image(img, str(file_name)[:-4] + ".png")

        accelerator.print('sampling complete')


    def cal_fid(self, target_path):
        command = 'fidelity -g 0 -f -i -b {} --input1 {} --input2 {}'\
            .format(self.batch_size, str(self.results_folder), target_path)
        os.system(command)

    def cal_metrics(self, task='cat2dog', source_gt_path=None, target_gt_path=None):
        from util.fid import calculate_fid_given_paths
        from util.mse_psnr_ssim_mssim import calculate_ssim, calculate_psnr, calculate_mse
        from util.eval_score import calculate_l2_given_paths
        import os
        
        translate_path = self.cfg.sampler.save_folder
        
        if self.accelerator.is_main_process:
            print(f"Calculating metrics for task: {task}")
            print(f"Translation path: {translate_path}")
            print(f"Source GT path: {source_gt_path}")
            print(f"Target GT path: {target_gt_path}")
            
            try:
                fid_value = calculate_fid_given_paths(paths=[translate_path, target_gt_path], dataset=task)
                print(f'FID: {fid_value}')
            except Exception as e:
                print(f"Failed to calculate FID: {e}")
                fid_value = None
            
            l2_distance = calculate_l2_given_paths(translate_path, source_gt_path)
            print(f'L2: {l2_distance}')
            
            mse = calculate_mse(translate_path, source_gt_path)
            print(f'MSE: {mse}')
            
            psnr_value = calculate_psnr(translate_path, source_gt_path)
            print(f'PSNR: {psnr_value}')
            
            ssim = calculate_ssim(translate_path, source_gt_path)
            print(f'SSIM: {ssim}')
            
            metrics_to_log = {}
            if fid_value is not None:
                metrics_to_log["evaluation/FID"] = fid_value
            metrics_to_log["evaluation/L2"] = l2_distance
            metrics_to_log["evaluation/MSE"] = mse
            metrics_to_log["evaluation/PSNR"] = psnr_value
            metrics_to_log["evaluation/SSIM"] = ssim
            
            swanlab.log(metrics_to_log)
            print("Evaluation metrics logged to SwanLab successfully.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass
