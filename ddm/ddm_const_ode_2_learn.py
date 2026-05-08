import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from .utils import default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, construct_class_by_name, safe_torch_load
from tqdm.auto import tqdm
from einops import rearrange, reduce
from functools import partial
from collections import namedtuple
from random import random, randint, sample, choice
from .encoder_decoder import DiagonalGaussianDistribution
import random
from taming.modules.losses.vqperceptual import *
from .augment import AugmentPipe
from .loss import *


# xt = x0 + C*t + t*\epsilon
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDPM(nn.Module):
    #============================================================1.准备阶段=============================================================
    def __init__(
        self,
        model,
        *,
        image_size,
        sampling_timesteps=None,
        loss_type='l2',
        objective='pred_noise',
        beta_schedule='cosine',
        clip_x_start=True,
        input_keys=['image'],
        start_dist='normal',
        sample_type='naive',
        perceptual_weight=1.,
        use_l1=False,
        **kwargs):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        cfg = kwargs.pop("cfg", None)
        super().__init__()
        # assert not (type(self) == DDPM and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.input_keys = input_keys
        self.cfg = cfg
        self.scale_input = self.cfg.get('scale_input', 1)
        self.register_buffer('eps', torch.tensor(
            cfg.get('eps', 1e-4) if cfg is not None else 1e-4))
        self.sigma_min = cfg.get(
            'sigma_min', 1e-2) if cfg is not None else 1e-2
        self.sigma_max = cfg.get('sigma_max', 1) if cfg is not None else 1
        self.weighting_loss = cfg.get(
            "weighting_loss", False) if cfg is not None else False
        if self.weighting_loss:
            print('#### WEIGHTING LOSS ####')

        self.clip_x_start = clip_x_start
        self.image_size = image_size
        self.objective = objective
        self.start_dist = start_dist
        assert start_dist in ['normal', 'uniform']

        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, 10)

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        loss_main_cfg_default = {'class_name': 'ddm.loss.MSE_Loss'}
        loss_vlb_cfg_default = {'class_name': 'ddm.loss.MAE_Loss'}
        loss_main_cfg = cfg.get('loss_main', loss_main_cfg_default)
        loss_vlb_cfg = cfg.get('loss_vlb', loss_vlb_cfg_default)
        self.loss_main_func = construct_class_by_name(**loss_main_cfg)
        self.loss_vlb_func = construct_class_by_name(**loss_vlb_cfg)
        self.use_l1 = use_l1

        self.perceptual_weight = perceptual_weight
        if self.perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()

        self.use_augment = self.cfg.get('use_augment', False)
        if self.use_augment:
            self.augment = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1,
                                       aniso=1, translate_frac=1)
            print('### use augment ###\n')

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False, use_ema=False):
        sd = safe_torch_load(path, map_location="cpu")
        if 'ema' in list(sd.keys()) and use_ema:
            sd = sd['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]    # remove ema_model.
                    new_sd[new_k] = sd[k]
                else:
                    new_sd[k] = sd[k]
            sd = new_sd
        else:
            if "model" in list(sd.keys()):
                sd = sd["model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        assert 'image' in self.input_keys
        if len(self.input_keys) > len(batch.keys()):
            x, *_ = batch.values()
        else:
            x = batch.values()
        return x

    #============================================================2.训练阶段============================================================
    def training_step(self, batch, *args, **kwargs):
        z, *_ = self.get_input(batch)
        cond = batch['cond'] if 'cond' in batch else None
        loss, loss_dict = self(z, cond)
        return loss, loss_dict

    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + time * noise
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - time * noise
        return x0

    def pred_xtms_from_xt(self, xt, noise, C, t, s):
        # noise = noise / noise.std(dim=[1, 2, 3]).reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        # mean = xt - C * s - (2*s*time - s**2) / time * noise
        # mean = xt - C * s - s * s * time / (s ** 2 + (time - s) ** 2) * noise
        mean = xt - C * s - (2*s*time - s**2) / time * noise
        epsilon = torch.randn_like(mean, dtype=torch.float64, device=xt.device)
        # sigma = (time-s) / time * torch.sqrt(2*s*time - s*s)
        # sigma = s * (time - s) / torch.sqrt(s ** 2 + (time - s) ** 2)
        sigma = torch.sqrt(2*s*time - s**2) * (time - s) / time
        xtms = mean + sigma * epsilon
        return xtms

    def p_losses(self, x_start, t, *args, **kwargs):
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        if self.use_augment:
            x_start, aug_label = self.augment(x_start)
            kwargs['augment_labels'] = aug_label
        # K = -1. * torch.ones_like(x_start)
        # C = noise - x_start  # t = 1000 / 1000
        C = -1 * x_start             # U(t) = Ct, U(1) = -x0
        x_noisy = self.q_sample(
            x_start=x_start, noise=noise, t=t, C=C)  # (b, c, h, w)
        pred = self.model(x_noisy, t, *args, **kwargs)
        C_pred, noise_pred = pred
        # C_pred = C_pred / torch.sqrt(t)
        # noise_pred = noise_pred / torch.sqrt(1 - t)
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t)
        # x_rec =
        loss_dict = {}
        prefix = 'train'
        target1 = C
        target2 = noise
        target3 = x_start
        loss_simple = 0.
        loss_vlb = 0.
        # use l1 + l2
        if self.weighting_loss:
            simple_weight1 = ((t - 1) / t) ** 2 + 1
            # simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps) ** 2 # eps prevents div 0
            simple_weight2 = (t / (1 - t + self.eps)) ** 2 + \
                1  # eps prevents div 0
        else:
            simple_weight1 = 1
            simple_weight2 = 1

        loss_simple += simple_weight1 * self.loss_main_func(C_pred, target1, reduction='sum') + \
            simple_weight2 * \
            self.loss_main_func(noise_pred, target2, reduction='sum')
        if self.use_l1:
            loss_simple += simple_weight1 * (C_pred - target1).abs().mean([1, 2, 3]) + \
                simple_weight2 * (noise_pred - target2).abs().mean([1, 2, 3])
            loss_simple = loss_simple / 2
        # rec_weight = 2 * (1 - t.reshape(C.shape[0], 1)) ** 2
        rec_weight = -torch.log(t.reshape(C.shape[0], 1)) / 2
        # loss_simple = loss_simple.sum() / C.shape[0]

        # loss_consist = torch.abs(x_noisy - self.q_sample(x_start=x_start, noise=noise_pred, t=t, C=C_pred)).mean([1, 2, 3])
        # loss_vlb += self.loss_vlb_func(x_rec, target3) * rec_weight ** 2
        # loss_vlb += loss_consist
        if self.perceptual_weight > 0.:
            loss_vlb += self.perceptual_loss(x_rec,
                                             target3).sum([1, 2, 3]) * rec_weight
        # loss_vlb = loss_vlb.sum() / C.shape[0]
        loss = loss_simple.sum() / C.shape[0] + loss_vlb.sum() / C.shape[0]
        loss_dict.update(
            {f'{prefix}/loss_simple': loss_simple.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})
        loss_dict.update(
            {f'{prefix}/loss_vlb': loss_vlb.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})
        loss_dict.update({f'{prefix}/loss': loss.detach().sum() /
                         C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(
                    target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
    
    def forward(self, x, *args, **kwargs):
        if self.scale_input != 1:
            x = x * self.scale_input
        # continuous time, t in [0, 1]
        eps = self.eps  # smallest time step
        t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        # t = torch.clamp(t, eps, 1)
        return self.p_losses(x, t, *args, **kwargs)
    
    #============================================================3.采样阶段===========================================================
    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, denoise=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        self.sample_type = self.cfg.get('sample_type', 'deterministic')
        if self.sample_type == 'deterministic':
            return self.sample_fn_d((batch_size, channels, image_size[0], image_size[1]),
                                    up_scale=up_scale, unnormalize=True, cond=cond, denoise=denoise)
        elif self.sample_type == 'stochastic':
            return self.sample_fn_s((batch_size, channels, image_size[0], image_size[1]),
                                    up_scale=up_scale, unnormalize=True, cond=cond, denoise=denoise)

    @torch.no_grad()
    def sample_fn_s(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        rho = 1
        step_indices = torch.arange(
            sampling_timesteps, dtype=torch.float64, device=device)
        t_steps = ((self.sigma_max ** 2) ** (1 / rho) + step_indices / (sampling_timesteps - 1) *
                   ((self.sigma_min ** 2) ** (1 / rho) - (self.sigma_max ** 2) ** (1 / rho))) ** rho
        # t_steps = t_steps * (1 / self.eps).round() * self.eps
        t_steps = torch.cat((t_steps, torch.tensor([0], device=device)), dim=0)
        time_steps = -torch.diff(t_steps)
        # step = 1. / self.sampling_timesteps
        # time_steps = torch.tensor([step], device=device).repeat(self.sampling_timesteps)
        # if denoise:
        #     eps = self.eps
        #     time_steps = torch.cat((time_steps[:-1], torch.tensor([time_steps[-1] - eps], device=device), \
        #                             torch.tensor([eps], device=device)), dim=0)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device, dtype=torch.float64)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        img = F.interpolate(img, scale_factor=up_scale,
                            mode='bilinear', align_corners=True) * self.sigma_max
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time)
            # C, noise = pred.chunk(2, dim=1)
            C, noise = pred[:2]
            # correct C
            x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            if self.clip_x_start:
                x0.clamp_(-1. * self.scale_input, 1. * self.scale_input)
            C = -1 * x0
            img = self.pred_xtms_from_xt(img, noise, C, cur_time, s)
            # img = self.pred_xtms_from_xt2(img, noise, C, cur_time, s)
            cur_time = cur_time - s
        img.clamp_(-1. * self.scale_input, 1. * self.scale_input)
        if self.scale_input != 1:
            img = img / self.scale_input
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample_fn_d(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        # step = self.sigma_min
        rho = 1.
        step_indices = torch.arange(
            sampling_timesteps, dtype=torch.float64, device=device)
        # t_steps = ((self.sigma_max ** 2) ** (1 / rho) + step_indices / (sampling_timesteps) * \
        #            ((self.sigma_min ** 2) ** (1 / rho) - (self.sigma_max ** 2) ** (1 / rho))) ** rho
        # t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
        #             self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
            step - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x_next = torch.randn(shape, device=device,
                             dtype=torch.float64) * t_steps[0]
        # img = F.interpolate(img, scale_factor=up_scale, mode='bilinear', align_corners=True)
        # cur_time = torch.ones((batch,), device=device)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # t_cur = torch.full((batch,), t_c, device=device)
            # t_next = torch.full((batch,), t_n, device=device)  # 0, ..., N-1
            x_cur = x_next
            if cond is not None:
                pred = self.model(x_cur, t_cur, cond)
            else:
                pred = self.model(x_cur, t_cur)
            C, noise = pred[:2]
            C, noise = C.to(torch.float64), noise.to(torch.float64)
            x0 = x_cur - C * t_cur - noise * t_cur
            # d_cur = (x_cur - x0) / t_cur
            # x_next = x_cur + (t_next - t_cur) * d_cur
            x_next = x0 + t_next * C + t_next * noise
            # d_cur = C + noise
            # x_next = x_cur + (t_next - t_cur) * d_cur
            # Apply 2-order correction.
            # if i < sampling_timesteps - 1:
            #     if cond is not None:
            #         pred = self.model(x_next, t_next, cond)
            #     else:
            #         pred = self.model(x_next, t_next)
            #     C_, noise_ = pred[:2]
            #     C_, noise_ = C_.to(torch.float64), noise_.to(torch.float64)
            #     d_next = C_ + noise_
            #     x_next = x_cur + (t_next - t_cur) * 0.5 * (d_cur + d_next)

        x_next.clamp_(-1. * self.scale_input, 1. * self.scale_input)
        if self.scale_input != 1:
            img = x_next / self.scale_input
        else:
            img = x_next
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img


class LatentDiffusion(DDPM):

    # =====================================================================1.准备阶段=========================================================================================

    def __init__(self,
                 auto_encoder,
                 scale_factor=1.0,
                 scale_by_std=True,
                 scale_by_softsign=False,
                 input_keys=['image'],
                 sample_type='naive',
                 default_scale=False,
                 *args,
                 **kwargs
                 ):
        self.scale_by_std = scale_by_std
        self.scale_by_softsign = scale_by_softsign
        self.default_scale = default_scale
        # self.perceptual_weight = 0
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        super().__init__(*args, **kwargs)
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        if self.scale_by_softsign:
            self.scale_by_std = False
            print('### USING SOFTSIGN RESCALING')
        assert (self.scale_by_std and self.scale_by_softsign) is False

        self.init_first_stage(auto_encoder)
        # self.instantiate_cond_stage(cond_stage_config)
        self.input_keys = input_keys
        self.clip_denoised = False
        # 'dpm' is not availible now, suggestion 'ddim'
        assert sample_type in ['naive', 'ddim', 'dpm', ]
        # self.sample_type = sample_type

        if self.cfg.get('use_disloss', False):
            loss_dis_func_default = {'class_name': 'ddm.loss.MAE_Loss'}
            loss_dis_func = self.cfg.get('loss_dis', loss_dis_func_default)
            self.loss_dis_func = construct_class_by_name(**loss_dis_func)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)
            
    # 初始化 Vae 模型 ，并冻结参数
    def init_first_stage(self, first_stage_model):
        self.first_stage_model = first_stage_model.eval()
        # self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
            
    '''
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = safe_torch_load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    '''
    
    # 自动计算和设置潜在空间的缩放因子 ， 如果参数default_scale设置为True，则不启动
    @torch.no_grad()
    def on_train_batch_start(self, batch):
        # only for the first batch
        if self.scale_by_std and (not self.scale_by_softsign):
            if not self.default_scale:
                assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
                # set rescale weight to 1./std of encodings
                print("### USING STD-RESCALING ###")
                x, *_ = batch.values()
                encoder_posterior = self.first_stage_model.encode(x)
                z = self.get_first_stage_encoding(encoder_posterior)
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())
                print(f"setting self.scale_factor to {self.scale_factor}")
                # print("### USING STD-RESCALING ###")
            else:
                print(f'### USING DEFAULT SCALE {self.scale_factor}')
        else:
            print(f'### USING SOFTSIGN SCALE !')

    # 负责将VAE编码器的输出转换为扩散模型可以使用的潜在表示，通过 detach() 确保两个阶段训练是解耦的
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # return self.scale_factor * z.detach() + self.scale_bias
        return z.detach()

    # =====================================================================2.训练阶段=========================================================================================
    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        assert 'image' in self.input_keys
        # if len(self.input_keys) > len(batch.keys()):
        #     x, cond, *_ = batch.values()
        # else:
        #     x, cond = batch.values()
        x = batch['image']
        cond = batch['cond'] if 'cond' in batch else None
        z = self.first_stage_model.encode(x)
        z = self.get_first_stage_encoding(z)
        # if self.cfg.get('use_disloss', False):
        out = [z, cond, x]
        # if return_first_stage_outputs:
        #     xrec = self.first_stage_model.decode(z)
        #     out.extend([x, xrec])
        # if return_original_cond:
        #     out.append(cond)
        return out

    def training_step(self, batch):
        z, c, x, *_ = self.get_input(batch)
        if self.scale_by_softsign:
            z = F.softsign(z)
        elif self.scale_by_std:
            z = self.scale_factor * z
        # print('grad', self.scale_bias.grad)
        if self.cfg.get('use_disloss', False):
            loss, loss_dict = self(z, c, ori_input=x)
        else:
            loss, loss_dict = self(z, c)
        return loss, loss_dict

    def p_losses(self, x_start, t, *args, **kwargs):
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        # K = -1. * torch.ones_like(x_start)
        # C = noise - x_start  # t = 1000 / 1000
        C = -1 * x_start             # U(t) = Ct, U(1) = -x0
        # C = -2 * x_start               # U(t) = 1/2 * C * t**2, U(1) = 1/2 * C = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, 2, c, h, w)
        pred = self.model(x_noisy, t, *args, **kwargs)
        C_pred, noise_pred = pred
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t)
        loss_dict = {}
        prefix = 'train'

        target1 = C
        target2 = noise
        target3 = x_start
        loss_simple = 0.
        loss_vlb = 0.
        # use l1 + l2
        if self.weighting_loss:
            simple_weight1 = ((t - 1) / t) ** 2 + 1
            # simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps) ** 2 # eps prevents div 0
            simple_weight2 = (t / (1 - t + self.eps)) ** 2 + \
                1  # eps prevents div 0
        else:
            simple_weight1 = 1
            simple_weight2 = 1

        loss_simple += simple_weight1 * self.loss_main_func(C_pred, target1, reduction='sum') + \
            simple_weight2 * \
            self.loss_main_func(noise_pred, target2, reduction='sum')
        
        if self.use_l1:
            loss_simple += simple_weight1 * (C_pred - target1).abs().sum([1, 2, 3]) + \
                simple_weight2 * (noise_pred - target2).abs().sum([1, 2, 3])
            loss_simple = loss_simple / 2
            
        loss = loss_simple.sum() / C.shape[0]
        
        rec_weight = -torch.log(t.reshape(C.shape[0], 1)) / 2
        # rec_weight = 2 * (1 - t.reshape(C.shape[0], 1)) ** 2
        loss_vlb += (x_rec - target3).abs().sum([1, 2, 3]) * rec_weight

        if self.cfg.get('use_disloss', False):
            with torch.no_grad():
                img_rec = self.first_stage_model.decode(
                    x_rec / self.scale_factor)
                img_rec = torch.clamp(
                    img_rec, min=-1., max=1.)  # B, 1, 320, 320
            loss_tmp = (img_rec - kwargs['ori_input']
                        ).sum([1, 2, 3]) * rec_weight  # B, 1
            if self.perceptual_weight > 0.:
                loss_tmp += self.perceptual_loss(
                    img_rec, kwargs['ori_input']).sum([1, 2, 3]) * rec_weight
            loss_distill = SpecifyGradient.apply(x_rec, loss_tmp.mean())
            loss_vlb += loss_distill  # .mean()

        loss += loss_vlb.sum() / C.shape[0]
        # loss = loss_simple.sum() / C.shape[0] + loss_vlb.sum() / C.shape[0]
        loss_dict.update(
            {f'{prefix}/loss_simple': loss_simple.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})
        loss_dict.update(
            {f'{prefix}/loss_vlb': loss_vlb.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})
        loss_dict.update(
            {f'{prefix}/loss': loss.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(
                    target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # =====================================================================3.采样/推理阶段=========================================================================================

    # 潜空间的确定性去噪
    @torch.no_grad()
    def sample_fn_d_c_list(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False, c_list=None):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        rho = 1.

        step_indices = torch.arange(
            sampling_timesteps, dtype=torch.float64, device=device)
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
            step - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        # x_next = torch.randn(shape, device=device, dtype=torch.float64) * t_steps[0]
        x_next = c_list.pop()

        for i, (t_cur, t_next, new_C) in enumerate(zip(t_steps[:-1], t_steps[1:], reversed(c_list))):
            # t_cur = torch.full((batch,), t_c, device=device)
            # t_next = torch.full((batch,), t_n, device=device)  # 0, ..., N-1
            x_cur = x_next
            if cond is not None:
                pred = self.model(x_cur, t_cur, cond)
            else:
                pred = self.model(x_cur, t_cur)
            C, noise = pred[:2]
            C, noise = new_C.to(torch.float64), noise.to(torch.float64)
            x0 = x_cur - C * t_cur - noise * t_cur
            x_next = x0 + t_next * C + t_next * noise
        img = x_next
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img
    
    # 完整的图像生成流程（图像确定性去噪+VAE解码） 
    @torch.no_grad()
    def sample_from_c_list(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True, c_list=None):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        down_ratio = self.first_stage_model.down_ratio
        self.sample_type = self.cfg.get('sample_type', 'deterministic')
        if self.sample_type == 'deterministic':
            z = self.sample_fn_d_c_list((batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio),
                                        up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise, c_list=c_list)

        if self.scale_by_std:
            z = 1. / self.scale_factor * z.detach()
        elif self.scale_by_softsign:
            z = z / (1 - z.abs())
            z = z.detach()
        # print(z.shape)
        x_rec = self.first_stage_model.decode(z.to(torch.float32))
        x_rec = unnormalize_to_zero_to_one(x_rec)
        x_rec = torch.clamp(x_rec, min=0., max=1.)
        if mask is not None:
            x_rec = mask * \
                unnormalize_to_zero_to_one(cond) + (1 - mask) * x_rec
        return x_rec

    @torch.no_grad()
    def sample_fn_s(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        # rho = 1.
        # step_indices = torch.arange(sampling_timesteps, dtype=torch.float32, device=device)
        # t_steps = ((self.sigma_max ** 2) ** (1 / rho) + step_indices / (sampling_timesteps) * \
        #            ((self.sigma_min ** 2) ** (1 / rho) - (self.sigma_max ** 2) ** (1 / rho))) ** rho
        # t_steps = torch.cat((t_steps, torch.tensor([0], device=device)), dim=0)
        # time_steps = -torch.diff(t_steps)
        step = 1. / self.sampling_timesteps
        time_steps = torch.tensor([step], dtype=torch.float32, device=device).repeat(
            self.sampling_timesteps)
        if denoise:
            eps = self.eps
            time_steps = torch.cat((time_steps[:-1], torch.tensor([time_steps[-1] - eps], device=device),
                                    torch.tensor([eps], device=device)), dim=0)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        img = F.interpolate(img, scale_factor=up_scale,
                            mode='bilinear', align_corners=True)
        # K = -1 * torch.ones_like(img)
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time)
            # C, noise = pred.chunk(2, dim=1)
            C, noise = pred[:2]
            if self.scale_by_softsign:
                # correct the C for softsign
                x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
                x0 = torch.clamp(x0, min=-0.987654321, max=0.987654321)
                C = -x0
            # correct C
            x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            C = -1 * x0
            img = self.pred_xtms_from_xt(img, noise, C, cur_time, s)
            # img = self.pred_xtms_from_xt2(img, noise, C, cur_time, s)
            cur_time = cur_time - s
        if self.scale_by_softsign:
            img.clamp_(-0.987654321, 0.987654321)
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample_fn_d(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        rho = 1.
        step_indices = torch.arange(
            sampling_timesteps, dtype=torch.float64, device=device)
        # t_steps = ((self.sigma_max ** 2) ** (1 / rho) + step_indices / (sampling_timesteps) * \
        #            ((self.sigma_min ** 2) ** (1 / rho) - (self.sigma_max ** 2) ** (1 / rho))) ** rho
        # t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
        #             self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
            step - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        # time_steps = -torch.diff(t_steps)
        # time_steps = torch.tensor([step], dtype=torch.float64, device=device).repeat(self.sampling_timesteps)
        # if denoise:
        #     eps = self.eps
        #     time_steps = torch.cat((time_steps[:-1], torch.tensor([time_steps[-1] - eps], device=device), \
        #                             torch.tensor([eps], dtype=torch.float64, device=device)), dim=0)

        x_next = torch.randn(shape, device=device,
                             dtype=torch.float64) * t_steps[0]
        # img = F.interpolate(img, scale_factor=up_scale, mode='bilinear', align_corners=True)
        # cur_time = torch.ones((batch,), device=device)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # t_cur = torch.full((batch,), t_c, device=device)
            # t_next = torch.full((batch,), t_n, device=device)  # 0, ..., N-1
            x_cur = x_next
            if cond is not None:
                pred = self.model(x_cur, t_cur, cond)
            else:
                pred = self.model(x_cur, t_cur)
            C, noise = pred[:2]
            C, noise = C.to(torch.float64), noise.to(torch.float64)
            x0 = x_cur - C * t_cur - noise * t_cur
            # d_cur = (x_cur - x0) / t_cur
            # x_next = x_cur + (t_next - t_cur) * d_cur
            x_next = x0 + t_next * C + t_next * noise
            # d_cur = C + noise
            # x_next = x_cur + (t_next - t_cur) * d_cur
            # Apply 2-order correction.
            # if i < sampling_timesteps - 1:
            #     if cond is not None:
            #         pred = self.model(x_next, t_next, cond)
            #     else:
            #         pred = self.model(x_next, t_next)
            #     C_, noise_ = pred[:2]
            #     C_, noise_ = C_.to(torch.float64), noise_.to(torch.float64)
            #     d_next = C_ + noise_
            #     x_next = x_cur + (t_next - t_cur) * 0.5 * (d_cur + d_next)

        img = x_next
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img
    
    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        down_ratio = self.first_stage_model.down_ratio
        self.sample_type = self.cfg.get('sample_type', 'deterministic')
        if self.sample_type == 'deterministic':
            z = self.sample_fn_d((batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio),
                                 up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise)
        elif self.sample_type == 'stochastic':
            z = self.sample_fn_s((batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio),
                                 up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise)

        if self.scale_by_std:
            z = 1. / self.scale_factor * z.detach()
        elif self.scale_by_softsign:
            z = z / (1 - z.abs())
            z = z.detach()
        # print(z.shape)
        x_rec = self.first_stage_model.decode(z.to(torch.float32))
        x_rec = unnormalize_to_zero_to_one(x_rec)
        x_rec = torch.clamp(x_rec, min=0., max=1.)
        if mask is not None:
            x_rec = mask * \
                unnormalize_to_zero_to_one(cond) + (1 - mask) * x_rec
        return x_rec
    
    # 扩散模型的推理函数 - 从图像提取图像分量列表
    # 
    # 【重要】c_list 的排序规则：
    #   c_list[0]  →  t ≈ step (小时间步) → 图像最清晰，包含精细细节
    #   c_list[1]  →  t 稍大              → 图像较清晰
    #   ...
    #   c_list[-1] →  t ≈ sigma_max (大时间步) → 图像最模糊，接近噪声
    #   
    #   即：c_list 从"清晰"到"模糊"排序，索引越小越清晰！
    #
    # 输入: x_start - 原始 RGB 图像 (已归一化到 [-1, 1])
    # 输出: 
    #   - c_list: 图像分量列表，每个元素 C = -x0 (预测原始图像的负值)
    #   - x_t: 最终的噪声状态 (在 t ≈ sigma_max 时刻)
    def reverse_q_sample_c_list_concat(self, x_start):
        device = x_start.device

        # ========== 第一步：VAE 编码 ==========
        z = self.first_stage_model.encode(x_start)
        z = self.get_first_stage_encoding(z)
        z = self.scale_factor * z
        x_t = z

        # ========== 第二步：构造时间步序列 ==========
        step = 1. / self.sampling_timesteps  # 例如 sampling_timesteps=20 时，step=0.05
        rho = 1.
        
        # 生成从 sigma_max 到 step 的递减序列
        step_indices = torch.arange(
            self.sampling_timesteps, dtype=torch.float32, device=device)
        t_steps = (self.sigma_max ** (1 / rho) + step_indices /
                   (self.sampling_timesteps - 1) * (step - self.sigma_max ** (1 / rho))) ** rho
        # 此时 t_steps = [sigma_max, ..., step]  (递减，例如 [1.00, 0.95, ..., 0.05])
        
        # 拼接 0 并反转，得到从小到大的时间步序列
        t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))
        # 最终 t_steps = [0, step, 2*step, ..., sigma_max]  (递增，例如 [0, 0.05, 0.10, ..., 1.00])
        # 
        # t_steps 数组结构:
        #   t_steps[0] = 0           (初始时刻，用于初始化)
        #   t_steps[1] = step        (第一个采样点)
        #   t_steps[2] = 2*step      (第二个采样点)
        #   ...
        #   t_steps[-2] = sigma_max-step (倒数第二个采样点)
        #   t_steps[-1] = sigma_max  (最大时间步)

        # ========== 第三步：初始化扩散过程 ==========
        # 在 t≈0 时刻预测初始分量和噪声（此 C 不加入 c_list）
        C, noise = self.model(x_t, t_steps[0].repeat(x_t.shape[0]) + 1e-4)
        # 正向加噪：将 x_t 从 t≈0 扩散到 t≈step+step
        x_t = self.q_sample(x_t, noise, t_steps[1].repeat(
            x_t.shape[0]) + step, C)

        c_list = []

        # ========== 第四步：循环提取图像分量 ==========
        # 使用 zip(t_steps[1:-1], t_steps[2:]) 构造相邻时间对:
        #   迭代 0: t_cur=t_steps[1]=step,     t_next=t_steps[2]=2*step     → c_list[0] (t最小，最清晰)
        #   迭代 1: t_cur=t_steps[2]=2*step,   t_next=t_steps[3]=3*step     → c_list[1]
        #   ...
        #   迭代 N-2: t_cur=t_steps[N-2],       t_next=t_steps[N-1]=sigma_max → c_list[N-2]
        for id, (t_cur, t_next) in enumerate(zip(t_steps[1:-1], t_steps[2:])):

            t_cur = t_cur.repeat(x_t.shape[0])
            # U-Net 预测当前时刻的 C 和 noise
            C, noise = self.model(x_t, t_cur)
            # 反解出预测的 x0 (原始图像估计)
            x0 = self.pred_x0_from_xt(x_t, noise, C, t_cur)
            # 提取图像分量: C = -x0
            C = -1 * x0
            # 正向加噪：将 x0 扩散到下一个更大的时间步
            x_t = self.q_sample(x0, noise, t_next.repeat(x_t.shape[0]), C)
            # 保存当前时间步的图像分量
            c_list.append(C)

        # ========== 第五步：处理最后一个时间步 ==========
        C, noise = self.model(x_t, t_next.repeat(x_t.shape[0]))
        x0 = self.pred_x0_from_xt(x_t, noise, C, t_next.repeat(x_t.shape[0]))
        C = -1 * x0
        c_list.append(C)

        return c_list, x_t


class EdgeLatentDiffusion(DDPM):
    """
    Edge-Preserving Decoupled Diffusion (EPDD) for Latent Space
    
    基于解耦扩散 ODE 框架的边缘保持潜空间扩散模型，核心改进：
    1. 在潜空间 z0 上计算梯度 ||∇z0|| 用于边缘检测
    2. 引入混合噪声方案（公式 EPDD-1, EPDD-2）
    3. 使用边缘感知加权损失（公式 EPDD-6，可选）
    4. 采样基于 ODE 解析解，与 LatentDiffusion 接口兼容
    """
    
    # =====================================================================1.准备阶段===================================================
    def __init__(self,
                 auto_encoder,
                 scale_factor=1.0,
                 scale_by_std=True,
                 scale_by_softsign=False,
                 input_keys=['image'],
                 sample_type='naive',
                 default_scale=False,
                 epdd_lambda_min=1e-5,
                 epdd_lambda_max=1e-1,
                 epdd_transition_point=0.5,
                 use_edge_weighted_loss=False,
                 *args,
                 **kwargs
                 ):
        self.scale_by_std = scale_by_std
        self.scale_by_softsign = scale_by_softsign
        self.default_scale = default_scale
        
        # EPDD 超参数
        self.epdd_lambda_min = float(epdd_lambda_min)
        self.epdd_lambda_max = float(epdd_lambda_max)
        self.epdd_transition_point = float(epdd_transition_point)
        self.use_edge_weighted_loss = use_edge_weighted_loss
        
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        super().__init__(*args, **kwargs)
        
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        if self.scale_by_softsign:
            self.scale_by_std = False
            print('### USING SOFTSIGN RESCALING')
        assert (self.scale_by_std and self.scale_by_softsign) is False

        self.init_first_stage(auto_encoder)
        self.input_keys = input_keys
        self.clip_denoised = False
        assert sample_type in ['naive', 'ddim', 'dpm', ]

        if self.cfg.get('use_disloss', False):
            loss_dis_func_default = {'class_name': 'ddm.loss.MAE_Loss'}
            loss_dis_func = self.cfg.get('loss_dis', loss_dis_func_default)
            self.loss_dis_func = construct_class_by_name(**loss_dis_func)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)
        
        print(f'### EPDD CONFIG: lambda_min={epdd_lambda_min}, lambda_max={epdd_lambda_max}, transition_point={epdd_transition_point}, use_edge_weighted_loss={use_edge_weighted_loss}')
    
    def init_first_stage(self, first_stage_model):
        """初始化 VAE 模型并冻结参数"""
        self.first_stage_model = first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def on_train_batch_start(self, batch):
        """自动计算和设置潜在空间的缩放因子"""
        if self.scale_by_std and (not self.scale_by_softsign):
            if not self.default_scale:
                assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
                print("### USING STD-RESCALING ###")
                x, *_ = batch.values()
                encoder_posterior = self.first_stage_model.encode(x)
                z = self.get_first_stage_encoding(encoder_posterior)
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())
                print(f"setting self.scale_factor to {self.scale_factor}")
            else:
                print(f'### USING DEFAULT SCALE {self.scale_factor}')
        else:
            print(f'### USING SOFTSIGN SCALE !')

    def get_first_stage_encoding(self, encoder_posterior):
        """将VAE编码器的输出转换为扩散模型可以使用的潜在表示"""
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return z.detach()
    
    # =====================================================================2.核心工具函数===============================================
    
    def compute_gradient_magnitude(self, z):
        """
        计算潜变量 z 的梯度幅值 ||∇z||
        使用 Sobel 算子，并对通道维度取均方根以保留边缘强度
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=z.dtype, device=z.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=z.dtype, device=z.device).view(1, 1, 3, 3)
        
        b, c, h, w = z.shape
        z_reshaped = z.view(b * c, 1, h, w)
        gx = F.conv2d(z_reshaped, sobel_x, padding=1).view(b, c, h, w)
        gy = F.conv2d(z_reshaped, sobel_y, padding=1).view(b, c, h, w)
        
        # 使用 RMS 聚合通道信息，比简单平均更能反映多通道的联合边缘
        grad_mag = torch.sqrt((gx**2 + gy**2 + 1e-8).mean(dim=1, keepdim=True))
        return grad_mag
    
    def compute_epdd_noise_coefficient(self, z0, t):
        """
        计算 EPDD 边缘保持噪声系数 σ_t^EP (公式 EPDD-2)
        
        Args:
            z0: 原始潜变量 [B, C, H, W]
            t: 时间步 [B] (归一化到 [0, 1])
        
        Returns:
            sigma_ep: 边缘保持噪声系数张量 [B, 1, H, W]
        """
        # 计算梯度幅值 ||∇z0||
        grad_mag = self.compute_gradient_magnitude(z0)  # [B, 1, H, W]
        
        # 计算时变边缘敏感度 λ(t)
        lambda_t = self.epdd_lambda_min + t.reshape(-1, 1, 1, 1) * (self.epdd_lambda_max - self.epdd_lambda_min)
        
        # 计算过渡函数 τ(t) - 线性过渡
        tau_t = torch.clamp(t / self.epdd_transition_point, 0.0, 1.0).reshape(-1, 1, 1, 1)
        
        # 公式 EPDD-2: σ_t^EP = 1 / [(1-τ(t)) * sqrt(1 + ||∇z0||/λ(t)) + τ(t)]
        denominator = (1 - tau_t) * torch.sqrt(1 + grad_mag / (lambda_t + 1e-8)) + tau_t
        sigma_ep = 1.0 / (denominator + 1e-8)
        
        return sigma_ep
    
    def compute_epdd_noise_coefficient_derivative(self, z0, t):
        """
        Compute dσ_t^EP/dt analytically
        
        Given σ_t^EP = 1/D(t) where D(t) = (1-τ(t))·A(t) + τ(t):
        - A(t) = sqrt(1 + |∇z0|/λ(t))
        - dD/dt = dτ/dt·(1-A) + (1-τ)·dA/dt
        - dA/dt = -|∇z0|·(λ_max-λ_min) / (2·λ²·A)
        - dσ/dt = -dD/dt / D²
        """
        grad_mag = self.compute_gradient_magnitude(z0)
        
        lambda_t = self.epdd_lambda_min + t.reshape(-1, 1, 1, 1) * (self.epdd_lambda_max - self.epdd_lambda_min)
        d_lambda_dt = self.epdd_lambda_max - self.epdd_lambda_min

        tau_t = torch.clamp(t / self.epdd_transition_point, 0.0, 1.0).reshape(-1, 1, 1, 1)
        d_tau_dt = torch.where(
            (t < self.epdd_transition_point).reshape(-1, 1, 1, 1),
            torch.ones_like(tau_t) / self.epdd_transition_point,
            torch.zeros_like(tau_t)
        )

        A_t = torch.sqrt(1 + grad_mag / (lambda_t + 1e-8))

        d_A_dt = -grad_mag * d_lambda_dt / (2 * (lambda_t + 1e-8)**2 * (A_t + 1e-8))

        D_t = (1 - tau_t) * A_t + tau_t

        d_D_dt = d_tau_dt * (1 - A_t) + (1 - tau_t) * d_A_dt

        d_sigma_dt = -d_D_dt / (D_t**2 + 1e-8)

        return d_sigma_dt

    def compute_effective_noise_coefficient(self, z0, t):
        """
        Compute the effective noise coefficient σ̃(t) = σ_t^EP + t·dσ_t^EP/dt
        
        This is the correct coefficient for the ODE velocity field:
        dx_t/dt = C + σ̃(t)·ε
        """
        sigma_ep = self.compute_epdd_noise_coefficient(z0, t)
        d_sigma_dt = self.compute_epdd_noise_coefficient_derivative(z0, t)
        time = t.reshape(-1, 1, 1, 1)
        sigma_eff = sigma_ep + time * d_sigma_dt
        return sigma_eff
    
    # =====================================================================3.训练阶段====================================================
    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        assert 'image' in self.input_keys
        x = batch['image']
        cond = batch['cond'] if 'cond' in batch else None
        z = self.first_stage_model.encode(x)
        z = self.get_first_stage_encoding(z)
        out = [z, cond, x]
        return out

    def training_step(self, batch):
        z, c, x, *_ = self.get_input(batch)
        if self.scale_by_softsign:
            z = F.softsign(z)
        elif self.scale_by_std:
            z = self.scale_factor * z
        
        if self.cfg.get('use_disloss', False):
            loss, loss_dict = self(z, c, ori_input=x)
        else:
            loss, loss_dict = self(z, c)
        return loss, loss_dict

    def q_sample(self, x_start, noise, t, C, sigma_ep=None):
        """
        EPDD 前向加噪过程 (公式 EPDD-1)
        x_t = (1-t)x_0 + t * σ_t^EP * ε
        
        当 sigma_ep=None 时，退化为标准解耦扩散:
        x_t = x_0 + C*t + t*ε = (1-t)x_0 + t*ε (因为 C=-x_0)
        
        Args:
            x_start: x0 [B, C, H, W]
            noise: ε ~ N(0, I) [B, C, H, W]
            t: 时间步 [B]
            C: 图像分量 C = -x0 (用于兼容基类接口)
            sigma_ep: 边缘保持噪声系数 σ_t^EP [B, 1, H, W]，如果为 None 则退化为标准扩散
        """
        time = t.reshape(-1, 1, 1, 1)
        
        if sigma_ep is not None:
            # EPDD 公式: x_t = (1-t)x_0 + t * σ_t^EP * ε
            x_noisy = (1 - time) * x_start + time * sigma_ep * noise
        else:
            # 标准解耦扩散: x_t = x_0 + C*t + t*ε = (1-t)x_0 + t*ε
            x_noisy = x_start + C * time + time * noise
        
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, C, t, sigma_ep=None):
        """
        从 x_t 预测 x_0
        
        由 EPDD 前向公式 x_t = (1-t)x_0 + t * σ_t^EP * ε 推导:
        x_0 = (x_t - t * σ_t^EP * ε) / (1-t)
        
        当 sigma_ep=None 时，退化为标准解耦扩散:
        x_0 = xt - C*t - t*noise
        
        参数签名与基类 DDPM 兼容: (xt, noise, C, t)，额外添加可选的 sigma_ep
        """
        time = t.reshape(-1, 1, 1, 1)
        
        if sigma_ep is not None:
            x0 = (xt - time * sigma_ep * noise) / (1 - time + 1e-8)
        else:
            # 标准解耦扩散反推
            x0 = xt - C * time - time * noise
        
        return x0

    def p_losses(self, x_start, t, *args, **kwargs):
        """
        EPDD 训练损失 (公式 EPDD-6)
        L_dm = E[||C_θ - C||² + ||ε_θ - ε||²]
        
        边缘加权损失为可选（由 use_edge_weighted_loss 控制）
        """
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        
        # 计算边缘保持噪声系数 σ_t^EP
        sigma_ep = self.compute_epdd_noise_coefficient(x_start, t)
        
        # 图像分量 C = -x0
        C = -1 * x_start
        
        # 前向加噪
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C, sigma_ep=sigma_ep)
        
        # 网络预测
        pred = self.model(x_noisy, t, *args, **kwargs)
        C_pred, noise_pred = pred
        
        # 从预测恢复 x0（修复：正确使用 pred_x0_from_xt 的签名）
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t, sigma_ep=sigma_ep)
        
        loss_dict = {}
        prefix = 'train'

        target1 = C
        target2 = noise
        target3 = x_start
        loss_simple = 0.
        loss_vlb = 0.
        
        # 损失权重
        if self.weighting_loss:
            simple_weight1 = ((t - 1) / t) ** 2 + 1
            simple_weight2 = (t / (1 - t + self.eps)) ** 2 + 1
        else:
            simple_weight1 = 1
            simple_weight2 = 1

        # 图像分量损失
        loss_simple += simple_weight1 * self.loss_main_func(C_pred, target1, reduction='sum')
        
        # 噪声损失（可选边缘加权）
        if self.use_edge_weighted_loss:
            # EPDD 边缘保持加权噪声损失
            # ||ε_θ - ε||²_Σ = Σ (1/σ_t^EP)² * (ε_θ - ε)²
            noise_diff = noise_pred - target2
            edge_weight = 1.0 / (sigma_ep ** 2 + 1e-8)
            # 截断权重防止梯度爆炸
            edge_weight = torch.clamp(edge_weight, max=1e4)
            weighted_noise_loss = (edge_weight * noise_diff ** 2).sum([1, 2, 3])
            loss_simple += simple_weight2 * weighted_noise_loss
        else:
            # 标准噪声损失（与 LatentDiffusion 一致）
            loss_simple += simple_weight2 * self.loss_main_func(noise_pred, target2, reduction='sum')
        
        if self.use_l1:
            loss_simple += simple_weight1 * (C_pred - target1).abs().sum([1, 2, 3])
            if self.use_edge_weighted_loss:
                noise_diff = noise_pred - target2
                edge_weight = 1.0 / (sigma_ep ** 2 + 1e-8)
                edge_weight = torch.clamp(edge_weight, max=1e4)
                loss_simple += simple_weight2 * (edge_weight.abs() * noise_diff.abs()).sum([1, 2, 3])
            else:
                loss_simple += simple_weight2 * (noise_pred - target2).abs().sum([1, 2, 3])
            loss_simple = loss_simple / 2
            
        loss = loss_simple.sum() / C.shape[0]
        
        # VLB 损失项（与 LatentDiffusion 一致）
        rec_weight = -torch.log(t.reshape(C.shape[0], 1)) / 2
        loss_vlb += (x_rec - target3).abs().sum([1, 2, 3]) * rec_weight

        if self.cfg.get('use_disloss', False):
            with torch.no_grad():
                img_rec = self.first_stage_model.decode(
                    x_rec / self.scale_factor)
                img_rec = torch.clamp(img_rec, min=-1., max=1.)
            loss_tmp = (img_rec - kwargs['ori_input']).sum([1, 2, 3]) * rec_weight
            if self.perceptual_weight > 0.:
                loss_tmp += self.perceptual_loss(
                    img_rec, kwargs['ori_input']).sum([1, 2, 3]) * rec_weight
            loss_distill = SpecifyGradient.apply(x_rec, loss_tmp.mean())
            loss_vlb += loss_distill

        loss += loss_vlb.sum() / C.shape[0]
        
        loss_dict.update(
            {f'{prefix}/loss_simple': loss_simple.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})
        loss_dict.update(
            {f'{prefix}/loss_vlb': loss_vlb.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})
        loss_dict.update(
            {f'{prefix}/loss': loss.detach().sum() / C.shape[0] / C.shape[1] / C.shape[2] / C.shape[3]})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss

    # =====================================================================4.采样/推理阶段===============================================
    
    @torch.no_grad()
    def sample_fn_d(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        """
        EPDD 确定性采样 (基于解耦扩散 ODE 解析解)
        
        使用与 LatentDiffusion 相同的时间步调度，但在更新时融入 sigma_ep
        ODE 更新: x_{t-s} = x_t - s*(C + σ_t^EP*ε)
        """
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        rho = 1.
        step_indices = torch.arange(
            sampling_timesteps, dtype=torch.float64, device=device)
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
            step - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x_next = torch.randn(shape, device=device,
                             dtype=torch.float64) * t_steps[0]
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if cond is not None:
                pred = self.model(x_cur, t_cur, cond)
            else:
                pred = self.model(x_cur, t_cur)
            C, noise = pred[:2]
            C, noise = C.to(torch.float64), noise.to(torch.float64)
            
            # 计算当前时刻的 sigma_ep（基于预测的 x0）
            # 先恢复 x0_pred
            x0_pred = x_cur - C * t_cur - noise * t_cur
            t_cur_tensor = torch.full((batch,), t_cur, device=device, dtype=torch.float64)
            sigma_eff = self.compute_effective_noise_coefficient(x0_pred.float(), t_cur_tensor.float()).to(torch.float64)
            
            s = t_cur - t_next
            x_next = x_cur - s * (C + sigma_eff * noise)
            
            # 等价于: x_next = x0_pred + t_next * C + t_next * sigma_ep * noise
            # 当 sigma_ep=1 时，退化为 LatentDiffusion 的更新
        
        img = x_next
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img
    
    @torch.no_grad()
    def sample_fn_s(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        """
        EPDD 随机性采样 (基于解耦扩散 ODE + 随机扰动)
        
        使用与 LatentDiffusion 相同的步长调度，融入 sigma_ep
        """
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        time_steps = torch.tensor([step], dtype=torch.float32, device=device).repeat(
            self.sampling_timesteps)
        if denoise:
            eps = self.eps
            time_steps = torch.cat((time_steps[:-1], torch.tensor([time_steps[-1] - eps], device=device),
                                    torch.tensor([eps], device=device)), dim=0)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        img = F.interpolate(img, scale_factor=up_scale,
                            mode='bilinear', align_corners=True)
        
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time)
            C, noise = pred[:2]
            
            if self.scale_by_softsign:
                x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
                x0 = torch.clamp(x0, min=-0.987654321, max=0.987654321)
                C = -x0
            
            # correct C
            x0 = self.pred_x0_from_xt(img, noise, C, cur_time)
            C = -1 * x0
            
            # 计算 sigma_ep
            sigma_eff = self.compute_effective_noise_coefficient(x0, cur_time)
            effective_noise = sigma_eff * noise
            img = self.pred_xtms_from_xt(img, effective_noise, C, cur_time, s)
            cur_time = cur_time - s
            
        if self.scale_by_softsign:
            img.clamp_(-0.987654321, 0.987654321)
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img
    
    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True):
        """完整的图像生成流程（潜空间确定性去噪+VAE解码）"""
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        down_ratio = self.first_stage_model.down_ratio
        self.sample_type = self.cfg.get('sample_type', 'deterministic')
        
        if self.sample_type == 'deterministic':
            z = self.sample_fn_d((batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio),
                                 up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise)
        elif self.sample_type == 'stochastic':
            z = self.sample_fn_s((batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio),
                                 up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise)

        if self.scale_by_std:
            z = 1. / self.scale_factor * z.detach()
        elif self.scale_by_softsign:
            z = z / (1 - z.abs())
            z = z.detach()
        
        x_rec = self.first_stage_model.decode(z.to(torch.float32))
        x_rec = unnormalize_to_zero_to_one(x_rec)
        x_rec = torch.clamp(x_rec, min=0., max=1.)
        if mask is not None:
            x_rec = mask * unnormalize_to_zero_to_one(cond) + (1 - mask) * x_rec
        return x_rec

    # =====================================================================5.逆向推理与结构提取 (适配 CycleDiff 循环训练)===============

    def reverse_q_sample_c_list_concat(self, x_start):
        """
        EPDD 逆向推理：从图像提取 C 列表
        使用与 LatentDiffusion 相同的时间步调度（非线性）
        """
        device = x_start.device

        # 1. VAE 编码与缩放
        z = self.first_stage_model.encode(x_start)
        z = self.get_first_stage_encoding(z)
        if self.scale_by_std:
            z = self.scale_factor * z
        elif self.scale_by_softsign:
            z = F.softsign(z)
        x_t = z

        # 2. 准备时间步 (与 LatentDiffusion 一致的非线性调度)
        step = 1. / self.sampling_timesteps
        rho = 1.
        step_indices = torch.arange(
            self.sampling_timesteps, dtype=torch.float32, device=device)
        t_steps = (self.sigma_max ** (1 / rho) + step_indices /
                   (self.sampling_timesteps - 1) * (step - self.sigma_max ** (1 / rho))) ** rho
        t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))

        # 3. 提取 t=0 时刻的「图像分量」和「噪声」
        C, noise = self.model(x_t, t_steps[0].repeat(x_t.shape[0]) + 1e-4)
        
        # 4. 正向加噪声 (EPDD 版本，使用 sigma_ep)
        t_next_val = t_steps[1].repeat(x_t.shape[0]) + step
        sigma_eff_0 = self.compute_effective_noise_coefficient(x_t, t_next_val)
        x_t = self.q_sample(x_t, noise, t_next_val, C=-x_t, sigma_ep=sigma_eff_0) 

        c_list = []

        # 5. 构造相邻时间对，并开始循环
        for id, (t_cur, t_next) in enumerate(zip(t_steps[1:-1], t_steps[2:])):
            t_cur = t_cur.repeat(x_t.shape[0])
            C, noise = self.model(x_t, t_cur)
            
            # 预测 x0（传入 sigma_ep 以正确反推）
            sigma_eff_cur = self.compute_effective_noise_coefficient(x_t, t_cur)
            x0 = self.pred_x0_from_xt(x_t, noise, C, t_cur, sigma_ep=sigma_eff_cur)
            C = -1 * x0
            
            # 再次正向加噪到下一步
            t_next_repeated = t_next.repeat(x_t.shape[0])
            sigma_eff_next = self.compute_effective_noise_coefficient(x0, t_next_repeated)
            x_t = self.q_sample(x0, noise, t_next_repeated, C=-x0, sigma_ep=sigma_eff_next)
            c_list.append(C)

        # 处理最后一个时间步
        t_last = t_steps[-1].repeat(x_t.shape[0])
        C, noise = self.model(x_t, t_last)
        sigma_eff_last = self.compute_effective_noise_coefficient(x_t, t_last)
        x0 = self.pred_x0_from_xt(x_t, noise, C, t_last, sigma_ep=sigma_eff_last)
        C = -1 * x0

        c_list.append(C)
        return c_list, x_t

    # =====================================================================6.高级采样接口 (适配 CycleDiff 工作流)========================
    
    @torch.no_grad()
    def sample_fn_d_c_list(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False, c_list=None):
        """
        EPDD 确定性采样（基于预计算的 C 列表 + ODE 解析解）
        与 LatentDiffusion 的 sample_fn_d_c_list 结构一致
        """
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        rho = 1.

        step_indices = torch.arange(
            sampling_timesteps, dtype=torch.float64, device=device)
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
            step - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        
        # 使用 C 列表的最后一个元素作为初始 x_next（与 LatentDiffusion 一致）
        x_next = c_list.pop()

        for i, (t_cur, t_next, new_C) in enumerate(zip(t_steps[:-1], t_steps[1:], reversed(c_list))):
            x_cur = x_next
            if cond is not None:
                pred = self.model(x_cur, t_cur, cond)
            else:
                pred = self.model(x_cur, t_cur)
            C, noise = pred[:2]
            C, noise = new_C.to(torch.float64), noise.to(torch.float64)
            
            t_cur_tensor = torch.full((batch,), t_cur, device=device, dtype=torch.float64)
            sigma_eff = self.compute_effective_noise_coefficient(x_cur.float(), t_cur_tensor.float()).to(torch.float64)
            
            s = t_cur - t_next
            x_next = x_cur - s * (C + sigma_eff * noise)
            
        img = x_next
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img
    
    @torch.no_grad()
    def sample_from_c_list(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True, c_list=None):
        """
        完整的图像生成流程（基于 C 列表的确定性去噪 + VAE解码）
        """
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        down_ratio = self.first_stage_model.down_ratio
        self.sample_type = self.cfg.get('sample_type', 'deterministic')
        
        if self.sample_type == 'deterministic':
            z = self.sample_fn_d_c_list(
                (batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio),
                up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise, c_list=c_list
            )
        elif self.sample_type == 'stochastic':
            # stochastic 模式下使用标准 sample_fn_s（不基于 C 列表）
            z = self.sample_fn_s(
                (batch_size, channels, image_size[0] // down_ratio, image_size[1] // down_ratio),
                up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise
            )

        if self.scale_by_std:
            z = 1. / self.scale_factor * z.detach()
        elif self.scale_by_softsign:
            z = z / (1 - z.abs())
            z = z.detach()
        
        x_rec = self.first_stage_model.decode(z.to(torch.float32))
        x_rec = unnormalize_to_zero_to_one(x_rec)
        x_rec = torch.clamp(x_rec, min=0., max=1.)
        if mask is not None:
            x_rec = mask * unnormalize_to_zero_to_one(cond) + (1 - mask) * x_rec
        return x_rec


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


if __name__ == "__main__":
    ddconfig = {'double_z': True,
                'z_channels': 4,
                'resolution': (240, 960),
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 2, 4, 4],  # num_down = len(ch_mult)-1
                'num_res_blocks': 2,
                'attn_resolutions': [],
                'dropout': 0.0}
    lossconfig = {'disc_start': 50001,
                  'kl_weight': 0.000001,
                  'disc_weight': 0.5}
    from encoder_decoder import AutoencoderKL
    auto_encoder = AutoencoderKL(ddconfig, lossconfig, embed_dim=4,
                                 )
    from mask_cond_unet import Unet
    unet = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=4, cond_in_dim=1,)
    ldm = LatentDiffusion(auto_encoder=auto_encoder,
                          model=unet, image_size=ddconfig['resolution'])
    image = torch.rand(1, 3, 128, 128)
    mask = torch.rand(1, 1, 128, 128)
    input = {'image': image, 'cond': mask}
    time = torch.tensor([1])
    with torch.no_grad():
        y = ldm.training_step(input)
    pass
