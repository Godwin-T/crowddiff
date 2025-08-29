import argparse
import os

import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util_v2 import TrainLoop, setup_dist_training
from guided_diffusion.resample import create_named_schedule_sampler

def main():
    args = create_argparser().parse_args()

    logger.configure(dir=args.log_dir)
    logger.log("creating model and diffusion...")

    # Define functions that will be passed to setup_dist_training
    def create_model():
        model, diffusion = sr_create_model_and_diffusion(
            **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
        )
        return model

    def create_diffusion():
        _, diffusion = sr_create_model_and_diffusion(
            **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
        )
        return diffusion

    def create_data():
        logger.log("creating data loader...")
        data = load_data(
            args.data_dir,
            args.batch_size,
            large_size=args.large_size,
            small_size=args.small_size,
            class_cond=args.class_cond,
        )
        return data

    def create_val_data():
        logger.log("creating validation data loader...")
        val_data = load_data(
            args.val_data_dir if args.val_data_dir else args.data_dir,
            args.val_batch_size,
            large_size=args.large_size,
            small_size=args.small_size,
            class_cond=args.class_cond,
            is_val=True,
        )
        return val_data

    def create_train_loop(model, diffusion, data, val_data):
        logger.log("creating training loop...")
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )
        return TrainLoop(
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
            multi_gpu=True,
        )

    # Create model and diffusion
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    
    # Define data loader functions that will be called in each process
    def create_data_loader():
        logger.log("creating data loader...")
        return load_data(
            args.data_dir,
            args.batch_size,
            large_size=args.large_size,
            small_size=args.small_size,
            class_cond=args.class_cond,
        )
    
    def create_val_data_loader():
        logger.log("creating validation data loader...")
        return load_data(
            args.val_data_dir if args.val_data_dir else args.data_dir,
            args.val_batch_size,
            large_size=args.large_size,
            small_size=args.small_size,
            class_cond=args.class_cond,
            is_val=True,
        )
    
    # Create the actual data loaders for the main process
    data = create_data_loader()
    val_data = create_val_data_loader()
    
    # Set environment variables for data parameters
    # These will be used by worker processes to recreate data loaders
    os.environ["DATA_DIR"] = args.data_dir
    os.environ["VAL_DATA_DIR"] = args.val_data_dir if args.val_data_dir else ""
    os.environ["LARGE_SIZE"] = str(args.large_size)
    os.environ["SMALL_SIZE"] = str(args.small_size)
    os.environ["CLASS_COND"] = "1" if args.class_cond else "0"
    
    # Create schedule sampler
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion
    )
    
    # Create TrainLoop instance
    logger.log("creating training loop...")
    train_loop = TrainLoop(
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
        multi_gpu=True,
    )

    # Launch distributed training
    setup_dist_training(train_loop)

def load_data(
    data_dir,
    batch_size,
    large_size,
    small_size,
    class_cond=False,
    is_val=False,
):
    """
    Create the data loader for the super-resolution model.
    This is a placeholder - replace with your actual data loading logic.
    """
    # Replace this with your actual data loading code
    from guided_diffusion.image_datasets import load_data
    return load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        is_val=is_val,
    )

def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
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
        large_size=256,
        small_size=64,
        class_cond=False,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
