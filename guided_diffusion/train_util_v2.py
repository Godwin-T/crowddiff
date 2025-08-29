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
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from torch.cuda.amp import autocast, GradScaler

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
        self.val_data=val_data
        self.normalizer=normalizer
        self.pred_channels=pred_channels
        self.base_samples=base_samples
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
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self.scaler = th.amp.GradScaler()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            # use_fp16=self.use_fp16,
            # fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.current_device = th.device(f'cuda:{self.rank}')
        self.model.to(self.current_device)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.model.parameters())
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            # Configure DDP model with appropriate settings for multi-GPU training
            if self.multi_gpu and th.cuda.device_count() > 1:
                # Get local rank from environment or use default device
                local_rank = self.rank
                th.cuda.set_device(local_rank)
                
                self.ddp_model = DDP(
                    self.model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
                logger.log(f"Using GPU {local_rank} with DDP, {th.cuda.device_count()} GPUs available")
            else:
                self.ddp_model = DDP(
                    self.model,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
                logger.log(f"Using single GPU with DDP")
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # set the resume to 0 to preclude importing the optimizer and ema model
            self.resume_step = 0
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                # self.model.load_state_dict(
                #     dist_util.load_state_dict(
                #         resume_checkpoint, map_location=dist_util.dev()
                #     ),strict=False
                # )
                checkpoint = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
                model_dict = self.model.state_dict()
                checkpoint = {k:v for k,v in checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
                model_dict.update(checkpoint)

                self.model.load_state_dict(model_dict)

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.model.parameters())

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.model.state_dict()

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
                opt_checkpoint, map_location=dist_util.dev()
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
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step %2000==0:
                try:
                    os.mkdir(os.path.join(self.log_dir,f'results_{self.step}'))
                except FileExistsError:
                    pass
                logger.log("creating samples...")
                all_images = []
                count=0
                while count  < 0:
                    count=count+1
                    model_kwargs = next(self.val_data)
                    name = model_kwargs['name'][0].split('.')[0]
                    del model_kwargs['name']
                    model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
                    crowd_den = th.clone(model_kwargs['high_res'])
                    del model_kwargs['high_res']
                    sample = self.diffusion.p_sample_loop(
                        self.model,
                        (1, self.pred_channels, model_kwargs['low_res'].shape[-2],model_kwargs['low_res'].shape[-1]),
                        model_kwargs=model_kwargs,
                    )

                    model_output, x0 = sample["sample"], sample["pred_xstart"]
                    sample = model_output.squeeze(0)
                    sample = [(item+1)*0.5 for item in sample]
                    sample = [item*255/(th.max(item)+1e-12) for item in sample]
                    sample = th.stack(sample).clamp(0,255).to(th.uint8)
                    sample = rearrange(sample, 'c h w -> h (c w)')
                    model_output = sample.contiguous().detach().cpu().numpy()

                    sample = x0.squeeze(0)
                    sample = [(item+1)*0.5 for item in sample]
                    sample = [item*255/(th.max(item)+1e-12) for item in sample]
                    sample = th.stack(sample).clamp(0,255).to(th.uint8)
                    sample = rearrange(sample, 'c h w -> h (c w)')
                    x0 = sample.contiguous().detach().cpu().numpy()

                    sample = np.concatenate([model_output, x0], axis=1)
                    sample = x0

                    crowd_den = crowd_den.squeeze(0)
                    crowd_den = [(item+1)*0.5*normalizer for item, normalizer in zip(crowd_den, self.normalizer)]
                    crowd_den = [item*255/(th.max(item)+1e-12) for item in crowd_den]
                    crowd_den = th.stack(crowd_den).clamp(0,255).to(th.uint8)
                    crowd_den = rearrange(crowd_den, 'c h w -> h (c w)')
                    crowd_den = crowd_den.contiguous().detach().cpu().numpy()

                    # req_image = np.concatenate([sample, crowd_den], axis=0)
                    req_image = [np.repeat(x[:,:,np.newaxis], 3, -1) for x in [sample, crowd_den]]
                    
                    crowd_img = model_kwargs["low_res"]
                    crowd_img = ((crowd_img + 1) * 127.5).clamp(0, 255).to(th.uint8)
                    crowd_img = crowd_img.permute(0, 2, 3, 1)
                    crowd_img = crowd_img.contiguous().cpu().numpy()[0]

                    # image = np.concatenate([crowd_img, np.zeros_like(crowd_img)], axis=0)
                    req_image = np.concatenate([req_image[0], crowd_img, req_image[-1]], axis=1)                    

                    if self.pred_channels == 1:
                        sample = np.repeat(sample,3,axis=-1)
                        crowd_den = np.repeat(crowd_den,3,axis=-1)

                    path = os.path.join(self.log_dir, f'results_{self.step}/{str(count)}.png')
                    cv2.imwrite(path, req_image[:,:,::-1])
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.model.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.model.zero_grad()
        print("*"*50)
        print(self.current_device)
        print("*"*50)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(self.current_device)
            micro_cond = {
                k: v[i : i + self.microbatch].to(self.current_device)
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
                print("*"*100)
                # if last_batch and not self.use_ddp:
                #     print("Condition True")
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
            self.scaler.scale(loss).backward()
            # self.mp_trainer.backward(scaler.scale(loss))

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate=rate)

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
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
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
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def extract_train_loop_args(train_loop):
    """Extract arguments from a TrainLoop instance as dictionaries that can be passed between processes"""
    # Extract model and diffusion separately
    model_args = {
        "model": train_loop.model,
    }
    
    diffusion_args = {
        "diffusion": train_loop.diffusion,
    }
    
    # Extract all other arguments
    train_loop_args = {
        "normalizer": train_loop.normalizer,
        "pred_channels": train_loop.pred_channels,
        "base_samples": train_loop.base_samples,
        "batch_size": train_loop.batch_size,
        "microbatch": train_loop.microbatch,
        "lr": train_loop.lr,
        "ema_rate": train_loop.ema_rate,
        "log_dir": train_loop.log_dir,
        "log_interval": train_loop.log_interval,
        "save_interval": train_loop.save_interval,
        "resume_checkpoint": train_loop.resume_checkpoint,
        "use_fp16": train_loop.use_fp16,
        "fp16_scale_growth": train_loop.fp16_scale_growth,
        "schedule_sampler": train_loop.schedule_sampler,
        "weight_decay": train_loop.weight_decay,
        "lr_anneal_steps": train_loop.lr_anneal_steps,
        "multi_gpu": True,
    }
    
    return model_args, diffusion_args, train_loop_args

def _distributed_worker(rank, world_size, model_args, diffusion_args, train_loop_args, data_dir, val_data_dir, 
                        batch_size, val_batch_size, large_size, small_size, class_cond):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        th.cuda.set_device(rank)
        logger.configure(dir=train_loop_args["log_dir"], rank=rank)
        
        # Create fresh data loaders in this process
        from guided_diffusion.image_datasets import load_data
        
        logger.log("creating data loader in worker process...")
        # data = load_data(
        #     data_dir=data_dir,
        #     batch_size=batch_size,
        #     image_size=large_size,
        #     class_cond=class_cond,
        #     is_val=False,
        # )
        
        logger.log("creating validation data loader in worker process...")
        # val_data = load_data(
        #     data_dir=val_data_dir if val_data_dir else data_dir,
        #     batch_size=val_batch_size,
        #     image_size=large_size,
        #     class_cond=class_cond,
        #     is_val=True,
        # )
        data = data_dir
        val_data = val_data_dir
        
        # Create a new TrainLoop for this process with the arguments
        worker_loop = TrainLoop(
            model=model_args["model"],
            diffusion=diffusion_args["diffusion"],
            data=data,
            val_data=val_data,
            **train_loop_args
        )
        
        worker_loop.run_loop()
        dist.destroy_process_group()
        
def setup_dist_training(train_loop):
    """
    Set up and launch distributed training across multiple GPUs.
    
    Args:
        train_loop: An instance of TrainLoop
    """
    if not th.cuda.is_available():
        logger.log("CUDA not available, running on CPU only")
        train_loop.run_loop()
        return
    
    # Check if we're already in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        logger.log("Using existing distributed environment")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        
        dist.init_process_group(
            backend="nccl", 
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        th.cuda.set_device(local_rank)
        logger.configure(dir=train_loop.log_dir, rank=rank)
        
        train_loop.run_loop()
        dist.destroy_process_group()
        return
    
    # Set up a new distributed environment
    n_gpus = th.cuda.device_count()
    if n_gpus > 1:
        logger.log(f"Launching distributed training on {n_gpus} GPUs")
        
        # Use torch.distributed.launch or torch.multiprocessing
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        
        # Extract arguments from train_loop to avoid pickling issues with generators
        model_args, diffusion_args, train_loop_args = extract_train_loop_args(train_loop)
        
        # Get data parameters from super_res_train_distributed.py
        # These will be used to recreate data loaders in each worker process
        from guided_diffusion.image_datasets import load_data
        
        # Get data directory and other parameters needed to recreate data loaders
        data_dir =train_loop.data
        val_data_dir = train_loop.val_data
        batch_size = train_loop.batch_size
        val_batch_size = 1  # Default value, adjust if needed
        large_size = int(os.environ.get("LARGE_SIZE", "256"))
        small_size = int(os.environ.get("SMALL_SIZE", "64"))
        class_cond = bool(int(os.environ.get("CLASS_COND", "0")))
        
        mp.spawn(
            _distributed_worker,
            args=(n_gpus, model_args, diffusion_args, train_loop_args, 
                  data_dir, val_data_dir, batch_size, val_batch_size, 
                  large_size, small_size, class_cond),
            nprocs=n_gpus,
            join=True
        )
    else:
        logger.log("Only one GPU available, running in single GPU mode")
        train_loop.run_loop()
