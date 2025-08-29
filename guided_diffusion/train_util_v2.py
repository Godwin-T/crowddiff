import copy
import functools
import os
import cv2
import numpy as np

from einops import rearrange

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from torch.amp import autocast, GradScaler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        val_data,
        normalizer,
        pred_channels,
        base_samples,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_dir,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        multi_gpu=True,
        rank: str = None
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.normalizer = normalizer
        self.pred_channels = pred_channels
        self.base_samples = base_samples
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.multi_gpu = multi_gpu
        self.rank = rank

        self.step = 0
        self.resume_step = 0

        # Determine the global batch size based on the number of GPUs
        if self.multi_gpu:
            self.global_batch = self.batch_size * dist.get_world_size()
            self.current_device = th.device(f'cuda:{self.rank}')
        else:
            self.global_batch = self.batch_size
            self.current_device = th.device(dist_util.dev())
        
        # Initialize GradScaler for automatic mixed precision training
        self.scaler = GradScaler()

        # Move model to the correct device and convert to half-precision if enabled
        self.model.to(self.current_device)
        # if self.use_fp16:
        #     self.model.half()

        # Create optimizer after moving the model to the device and converting its dtype
        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Handle distributed data parallel setup
        if self.multi_gpu and th.cuda.is_available() and th.cuda.device_count() > 1:
            self.use_ddp = True
            th.cuda.set_device(self.current_device)
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.current_device],
                output_device=self.current_device,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            if self.use_fp16:
                self.ddp_model.module.half() 
                # Ensure trainable parameters are in float32 for the GradScaler
                for param in self.ddp_model.parameters():
                    if param.requires_grad:
                        param.data = param.data.float()
            logger.log(f"Using GPU {self.rank} with DDP, {th.cuda.device_count()} GPUs available")
        else:
            self.use_ddp = False
            self.ddp_model = self.model
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. Gradients will not be synchronized properly!"
                )
        
        
        logger.log(f"Model parameters and buffers verified on device cuda:{rank}")
        # Load and sync parameters for resuming training
        self._load_and_sync_parameters()

        # Initialize EMA parameters
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                [p.detach().clone().cpu() for p in self.model.parameters()]
                for _ in range(len(self.ema_rate))
            ]
        else:
            # Create a deep copy of model parameters for EMA
            self.ema_params = [
                [p.detach().clone().cpu() for p in self.model.parameters()]
                for _ in range(len(self.ema_rate))
            ]

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # set the resume to 0 to preclude importing the optimizer and ema model
            self.resume_step = 0
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                checkpoint = dist_util.load_state_dict(resume_checkpoint, map_location=self.current_device)
                model_dict = self.model.state_dict()
                checkpoint = {k:v for k,v in checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
                model_dict.update(checkpoint)
                self.model.load_state_dict(model_dict)
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(list(p.cpu() for p in self.model.parameters()))

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=self.current_device
                )
                ema_params = list(state_dict.values())
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=self.current_device
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step % 2000 == 0:
                try:
                    os.mkdir(os.path.join(self.log_dir, f'results_{self.step}'))
                except FileExistsError:
                    pass
                logger.log("creating samples...")
                all_images = []
                count = 0
                while count < 0:
                    count += 1
                    model_kwargs = next(self.val_data)
                    name = model_kwargs['name'][0].split('.')[0]
                    del model_kwargs['name']
                    model_kwargs = {k: v.to(self.current_device) for k, v in model_kwargs.items()}
                    crowd_den = th.clone(model_kwargs['high_res'])
                    del model_kwargs['high_res']
                    
                    sample = self.diffusion.p_sample_loop(
                        self.model,
                        (1, self.pred_channels, model_kwargs['low_res'].shape[-2], model_kwargs['low_res'].shape[-1]),
                        model_kwargs=model_kwargs,
                    )

                    model_output, x0 = sample["sample"], sample["pred_xstart"]
                    sample = model_output.squeeze(0)
                    sample = [(item + 1) * 0.5 for item in sample]
                    sample = [item * 255 / (th.max(item) + 1e-12) for item in sample]
                    sample = th.stack(sample).clamp(0, 255).to(th.uint8)
                    sample = rearrange(sample, 'c h w -> h (c w)')
                    model_output = sample.contiguous().detach().cpu().numpy()

                    sample = x0.squeeze(0)
                    sample = [(item + 1) * 0.5 for item in sample]
                    sample = [item * 255 / (th.max(item) + 1e-12) for item in sample]
                    sample = th.stack(sample).clamp(0, 255).to(th.uint8)
                    sample = rearrange(sample, 'c h w -> h (c w)')
                    x0 = sample.contiguous().detach().cpu().numpy()

                    sample = np.concatenate([model_output, x0], axis=1)
                    sample = x0

                    crowd_den = crowd_den.squeeze(0)
                    crowd_den = [(item + 1) * 0.5 * normalizer for item, normalizer in zip(crowd_den, self.normalizer)]
                    crowd_den = [item * 255 / (th.max(item) + 1e-12) for item in crowd_den]
                    crowd_den = th.stack(crowd_den).clamp(0, 255).to(th.uint8)
                    crowd_den = rearrange(crowd_den, 'c h w -> h (c w)')
                    crowd_den = crowd_den.contiguous().detach().cpu().numpy()

                    req_image = [np.repeat(x[:, :, np.newaxis], 3, -1) for x in [sample, crowd_den]]

                    crowd_img = model_kwargs["low_res"]
                    crowd_img = ((crowd_img + 1) * 127.5).clamp(0, 255).to(th.uint8)
                    crowd_img = crowd_img.permute(0, 2, 3, 1)
                    crowd_img = crowd_img.contiguous().cpu().numpy()[0]

                    req_image = np.concatenate([req_image[0], crowd_img, req_image[-1]], axis=1)

                    if self.pred_channels == 1:
                        sample = np.repeat(sample, 3, axis=-1)
                        crowd_den = np.repeat(crowd_den, 3, axis=-1)

                    path = os.path.join(self.log_dir, f'results_{self.step}/{str(count)}.png')
                    cv2.imwrite(path, req_image[:, :, ::-1])
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        # Use scaler for optimization
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()
        # self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i:i + self.microbatch].to(self.current_device)
            # micro_cond = {
            #     k: v[i:i + self.microbatch].to(self.current_device)
            #     for k, v in cond.items()
            # }
            micro = batch[i:i + self.microbatch].to(self.current_device, dtype=th.float32)
            micro_cond = {
                k: v[i:i + self.microbatch].to(self.current_device, dtype=th.float32)
                for k, v in cond.items()
            }
            with th.amp.autocast(enabled=self.use_fp16, device_type="cuda"):
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], self.current_device)

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch and not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )

            # Use scaler for backward pass
            self.scaler.scale(loss).backward()


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            with th.no_grad():
                for param, ema_param in zip(self.model.parameters(), params):
                    ema_param.copy_(param.detach().lerp(ema_param, rate))

    # def _update_ema(self):
    #     """
    #     Updates the EMA model parameters using an exponential moving average.
    #     """
    #     for param, ema_param in zip(self.model.parameters(), self.ema_parameters()):
    #         # Ensure both tensors are on the same device before the operation
    #         if param.device != ema_param.device:
    #             # Forcing ema_param to the same device as param
    #             ema_param.data = ema_param.data.to(param.device)

    #         # The rest of the update is the same
    #         ema_param.copy_(param.detach().lerp(ema_param, self.ema_rate))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                    state_dict = self.model.state_dict()
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                    state_dict = {
                        "state_dict": {
                            "module." + k: v
                            for k, v in zip(self.model.state_dict().keys(), params)
                        }
                    }
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
                logger.log(f"Model successfully saved to {filename}...")

        save_checkpoint(0, None)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)