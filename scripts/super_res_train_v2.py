"""
Train a super-resolution model.
"""

import sys
import os

# Path to the folder containing your modules
module_path = os.path.abspath("/kaggle/working/crowddiff/guided_diffusion")

# Add to sys.path if not already there
if module_path not in sys.path:
    sys.path.append(module_path)
    
import argparse
import glob
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch as th

from time import time, sleep

import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util_v2 import TrainLoop, setup_dist_training


def main():
    
    
    args = create_argparser().parse_args()
    if args.multi_gpu:
        # Use all available GPUs
        n_gpus = th.cuda.device_count()
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs for distributed training")
            mp.spawn(train_worker, args=(n_gpus, args), nprocs=n_gpus, join=True)
        else:
            print("Only one GPU available, running in single GPU mode")
            train_worker(0, 1, args)
    else:
        # Original single GPU code path
        dist_util.setup_dist()
        logger.configure(dir=args.log_dir)#, format_strs=['stdout', 'wandb'])
        run_training(args)

def train_worker(rank, world_size, args):
    """Per-process training function for multi-GPU training"""
    # Setup process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Configure device and logger
    th.cuda.set_device(rank)
    # logger.configure(dir=args.log_dir, rank=rank)
    print("*"*50)
    print("Device Ranks")
    print(rank)
    print("*"*50)
    
    run_training(args, rank)
    
    # Clean up process group
    dist.destroy_process_group()

def run_training(args, rank):
    """Main training logic, separated to be called from either single or multi-GPU paths"""

    logger.log("creating model...")

    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )

    # model.to(dist_util.dev())
    if args.use_fp16:
      model.half()
    
    # Wrap model with DistributedDataParallel for multi-GPU training
    # if dist.is_initialized() and dist.get_world_size() > 1:
    #     model = th.nn.parallel.DistributedDataParallel(
    #         model, 
    #         device_ids=[dist.get_rank()],
    #         output_device=dist.get_rank(),
    #         broadcast_buffers=False,
    #         find_unused_parameters=True
    #     )
    #     logger.log(f"Model wrapped with DistributedDataParallel on rank {dist.get_rank()}")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    args.normalizer = [float(value) for value in args.normalizer.split(',')]
    # args.num_classes = [str(index) for index in range(args.num_classes)]
    # args.num_classes = sorted(args.num_classes)
    # args.num_classes = {k: i for i,k in enumerate(args.num_classes)}

    logger.log("creating data loader...")
    data = load_superres_data(
        data_dir = args.data_dir,
        batch_size = args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        normalizer=args.normalizer,
        pred_channels=args.pred_channels,
    )
    # val_data = load_data_for_worker(args.val_samples_dir,args.val_batch_size, args.normalizer, args.pred_channels,
    #                                 args.num_classes, class_cond=True)
    val_data = load_data_for_worker(args)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        val_data=val_data,
        normalizer=args.normalizer,
        pred_channels=args.pred_channels,
        base_samples=args.val_samples_dir,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        rank=rank
    ).run_loop()
    # setup_dist_training(train_loop=training_loop)


def load_superres_data(data_dir, batch_size, large_size, small_size, normalizer, pred_channels, class_cond=False):
    print("Data Directory")
    print(data_dir)
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        normalizer=normalizer,
        pred_channels=pred_channels,
    )
    for large_batch, model_kwargs in data:
        # model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        large_batch, model_kwargs["low_res"] = large_batch[:,:pred_channels], large_batch[:,pred_channels:]
        yield large_batch, model_kwargs


# def load_data_for_worker(base_samples, batch_size, normalizer, pred_channels, class_cond=False):
def load_data_for_worker(args):
    base_samples, batch_size, normalizer, pred_channels = args.val_samples_dir, args.val_batch_size, args.normalizer, args.pred_channels
    class_labels, class_cond = args.num_classes, args.class_cond
    # start = time()
    img_list = glob.glob(os.path.join(base_samples,'*.jpg'))
    img_list = img_list
    den_list = []
    for _ in img_list:
        den_path =  _.replace('test','test_den')
        den_path = den_path.replace('.jpg','.csv')
        den_list.append(den_path)
    # print(f'list prepared: {(time()-start) :.4f}s.')

    image_arr, den_arr = [], []
    for file in img_list:
        # start = time()
        image = Image.open(file)
        image_arr.append(np.asarray(image))
        # print(f'image read: {(time()-start) :.4f}s.')

        # start = time()
        file = file.replace('test','test_den').replace('jpg','csv')
        image = np.asarray(pd.read_csv(file, header=None).values)
        # print(f'density read: {(time()-start) :.4f}s.')

        # start = time()
        image = np.stack(np.split(image, len(normalizer), -1))
        image = np.asarray([m/n for m,n in zip(image, normalizer)])
        image = image.transpose(1,2,0).clip(0,1)
        den_arr.append(image)
        # print(f'density prepared: {(time()-start) :.4f}s.')


    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer, den_buffer = [], []
    label_buffer = []
    name_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i]), den_buffer.append(den_arr[i])
            name_buffer.append(os.path.basename(img_list[i]))
            if class_cond:
                class_label = os.path.basename(img_list[i]).split('_')[0]
                class_label = class_labels[class_label]
                label_buffer.append(class_label)
                # pass
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                den_batch = th.from_numpy(np.stack(den_buffer)).float()
                # den_batch = den_batch / normalizer 
                den_batch = 2*den_batch - 1
                den_batch = den_batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch,
                           name=name_buffer,
                           high_res=den_batch
                           )
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer, name_buffer, den_buffer = [], [], [], []


def create_argparser():
    defaults = dict(
        data_dir="",
        val_batch_size=1,
        val_samples_dir=None,
        log_dir=None,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        normalizer='0.2',
        pred_channels=3,
        num_classes=13,
        multi_gpu=True,  # Enable multi-GPU training by default
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # run_training(args)
    main()
