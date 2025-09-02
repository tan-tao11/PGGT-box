import os
import os.path as osp
import shutil
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import random

from omegaconf import OmegaConf
from datetime import timedelta
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from lion_pytorch import Lion

from vggt.models.vggt import VGGT
from utils.load_model import load_model
from src.data.onepose import Onepose
from src.data.gso import Gso
from src.losses.loss import cal_loss
from src.losses.loss_with_conf import compute_camera_loss
from validation import validate_model, compute_metrics, vis
from utils.comman_utils import generate_uuid_string


def train_worker(gpu_id: int, config: OmegaConf):
    """Train diffusion model on the specified GPU."""
    # Initialize the process group for distributed training
    world_size = config.gpus
    dist.init_process_group(
        backend="nccl", timeout=timedelta(seconds=7200), rank=gpu_id, world_size=world_size
    )

    # Set the device for the current process
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Load the model and other components based on the config
    model = load_model(config, device=device)
    # from torchinfo import summary

    # summary(model, input_size=(1, 17, 3, 224, 224))  # 输入张量的形状
    # exit()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False
    )

    # Load dataset and data loader
    print("Loading OnePose dataset...")
    OnePose_data = Onepose(config)
    print("Loading Gso dataset...")
    Gso_data = Gso(config, device)
    # repeat_factor = len(Gso_data) // len(OnePose_data)
    # if repeat_factor > 1:
    #     OnePose_data = ConcatDataset([OnePose_data] * repeat_factor)
    # train_dataset = ConcatDataset([OnePose_data, Gso_data])
    # train_dataset = Gso_data
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    # train_dataset, rank=gpu_id, shuffle=False
    # )
    # train_data_loader = torch.utils.data.DataLoader(
    #     dataset = train_dataset,
    #     batch_size=config.train.batch_size,
    #     shuffle=False,
    #     num_workers=config.train.num_workers,
    #     pin_memory=True,
    #     sampler=train_sampler,
    #     persistent_workers=False,
    # )

    train_dataset = OnePose_data
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=gpu_id, shuffle=True)
    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=False,
    )
    train_dataset1 = Gso_data
    train_sampler1 = torch.utils.data.distributed.DistributedSampler(train_dataset1, rank=gpu_id, shuffle=True)
    train_data_loader1 = torch.utils.data.DataLoader(
        dataset = train_dataset1,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        sampler=train_sampler1,
        persistent_workers=False,
    )
    data1_iter = iter(train_data_loader1)

    # Initialize the optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
       # --- 从这里开始修改 ---

    # Initialize the optimizer
    # 建议将学习率和warmup步数也放入config文件中，这里为了演示直接写出
    target_lr = config.train.lr  # 建议使用一个降低后的目标学习率
    warmup_steps = config.train.warmup_steps # 假设预热1000步
    optimizer = torch.optim.AdamW(model.parameters(), lr=target_lr, weight_decay=1e-2)
    # optimizer = Lion(model.parameters(), lr=target_lr, weight_decay=0.05)

    max_epochs = config.train.max_epochs
    total_steps = max_epochs * len(train_data_loader)
    
    # 1. 创建用于 Warm-up 的调度器
    # 在 warmup_steps 步内，学习率从 target_lr * 0.01 线性增长到 target_lr * 1
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps
    )

    # 2. 创建主调度器（CosineAnnealing）
    # 注意：T_max 需要减去 warm-up 的步数
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )

    # 3. 使用 SequentialLR 将两者串联起来
    # 在前 warmup_steps 步，使用 warmup_scheduler
    # 在 warmup_steps 步之后，切换到 main_scheduler
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
    )

    # Training loop setup
    model.train()
    step = 0

    if gpu_id == 0:
        progress_bar = tqdm(total=total_steps, desc="Training Progress")
        
        # Initialize Tensorboard writer
        tb_root = osp.join(config.train.log_dir, "tensorboard")

        if osp.isdir(tb_root):
            contents = os.listdir(tb_root)
            if contents:
                os.makedirs(tb_root + "_backup", exist_ok=True)

                for content in contents:
                    shutil.move(osp.join(tb_root, content), osp.join(tb_root + "_backup", content+generate_uuid_string(4)))
        
        tb_folder = osp.join(tb_root, f"run_{config.run_name}")
        os.makedirs(tb_folder, exist_ok=True)
        print(f"Tensorboard logs will be saved to {tb_folder}")

        writer = SummaryWriter(log_dir=tb_folder)

        # model.eval()
        # loss_dict, rotation_angle_error, translation_error, ratio, vis_images = validate_model(model.module, config, device)
        # for key, value in loss_dict.items():
        #     writer.add_scalar(f'Val_loss/{key}', value, step)
        # writer.add_scalar('Rotation Angle Error/val', rotation_angle_error, step)
        # writer.add_scalar('Translation Error/val', translation_error, step)
        # writer.add_scalar('Ratio/val', ratio, step)
        # if len(vis_images) > 0:
        #     for idx, image in enumerate(vis_images):
        #         writer.add_image(f'Val_images/{idx}', image, step)
        # model.train()
    dist.barrier()

    # Training loop
    print("Starting training...")
    # Initialize the scaler
    scaler = torch.cuda.amp.GradScaler()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    for epoch in range(max_epochs):
        train_sampler.set_epoch(epoch)  # Set the epoch for the sampler
        train_sampler1.set_epoch(epoch)
        for _, data in enumerate(train_data_loader):
            # Wether to use synthetic data
            if random.random() < 0.5:
                data = next(data1_iter)
            # Zero the gradients
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=dtype):
                # Forward pass
                pred = model(data, device)
                # with torch.no_grad():
                #     pred = model(data, device)
            
            loss, loss_dict = compute_camera_loss(pred[0], pred[1], data, config=config)

            # Scale the loss
            scaler.scale(loss).backward()
            
            # ✅ 在 step 之前、unscale 后裁剪梯度
            scaler.unscale_(optimizer)  # 反缩放梯度，使得 clip_grad_norm_ 有效
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            # Update learning rate
            scheduler.step()

            # Print progress
            if gpu_id == 0:
                step += 1
                progress_bar.update(1)
                progress_bar.set_description(
                    f'Epoch={epoch}/{max_epochs} LR={scheduler.get_last_lr()[0]:.2e} Loss={loss.item():.4f}'
                )

                if step % config.train.save_itr == 0:
                    save_model(config, step, f'{step:06d}', model, optimizer, scheduler)

                for key, value in loss_dict.items():
                    writer.add_scalar(f'Train_loss/{key}', value.item(), step)

                if step % config.train.metric_period == 0:
                    query_info = data['query_info']
                    ref_info = data['ref_info']
                    rotation_angle_error, translation_error, query_bbox, gt_query_bbox, ref_bbox, ref_indice = compute_metrics(pred[0][-1], pred[1][-1], query_info, ref_info, config)
                    vis_query_image = vis(data['images'][0, 0], query_bbox, gt_query_bbox, config)
                    vis_ref_image = vis(data['images'][0, ref_indice+1], ref_bbox, ref_bbox, config)
                    writer.add_scalar('Rotation Angle Error/train', rotation_angle_error, step)
                    writer.add_scalar('Translation Error/train', translation_error, step)
                    writer.add_image('Train_images/query', vis_query_image, step)   
                    writer.add_image('Train_images/ref', vis_ref_image, step)


            if gpu_id == 0 and step % config.train.val_period == 0:
                # Validate model
                model.eval()
                loss_dict, rotation_angle_error, translation_error, ratio, vis_images = validate_model(model.module, config, device)
                # writer.add_scalar('Loss/val_trans', val_loss_trans, step)
                # writer.add_scalar('Loss/val_rot', val_loss_rot, step)
                for key, value in loss_dict.items():
                    writer.add_scalar(f'Val_loss/{key}', value, step)
                writer.add_scalar('Rotation Angle Error/val', rotation_angle_error, step)
                writer.add_scalar('Translation Error/val', translation_error, step)
                writer.add_scalar('Ratio/val', ratio, step)
                if len(vis_images) > 0:
                    for idx, image in enumerate(vis_images):
                        writer.add_image(f'Val_images/{idx}', image, step)
                model.train()
            dist.barrier()
                
    # Finalize training and save model                
    if gpu_id == 0:
        save_model(config, step, f'{step:06d}', model, optimizer, scheduler)


def save_model(cfg, step, mod, model, optimizer, scheduler):
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f'{cfg.save_dir}/ckpt_{mod}.pt')

# Test train_worker
# if __name__ == "__main__":
#     config_path = 'configs/train.yaml'
    
#     config = OmegaConf.load(config_path)
#     train_worker(gpu_id=0, config=config)
