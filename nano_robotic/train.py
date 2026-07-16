"""
Main training entrypoint for the VLA Foundry training.

This module wires together configuration parsing, model/optimizer/scheduler
construction, dataset selection per checkpoint, and the core training loop.

Notes
- The training budget is specified in *samples*, not *steps*.
- This file aims to remain a *thin orchestrator*; most heavy lifting is
delegated to subpackages (data, models, opt, train, etc.).
"""

import torch
import argparse
import os

import itertools
import logging
import time
from collections.abc import Callable
from torch.optim.optimizer import Optimizer

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from nano_robotic.modules.vla import VLA

from nano_robotic.data.dataloader import get_datastring_input, get_dataloader
from nano_robotic.utils.optimizer import create_optimizer
from nano_robotic.utils.lr_scheduler import lr_scheduler
from nano_robotic.utils.loss_function import masked_mse_loss
from nano_robotic.utils.utils import set_random_seed, load_yaml, count_parameters, print0, world_info_from_env
from nano_robotic.utils.file_utils import save_checkpoint
from nano_robotic.batch_handlers import DiffusionPolicyBatchHandler

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain base model")    
    
    # training dataset
    parser.add_argument("--camera_names", type=str, default="['exterior_image_1_left','exterior_image_2_left','wrist_image_left']", help="")
    parser.add_argument("--action_fields", type=str, default="['action']", help="")
    parser.add_argument("--proprioception_fields", type=str, default="['observation.state']", help="")
    parser.add_argument("--lowdim_past_timesteps", type=int, default=2, help="")
    parser.add_argument("--lowdim_future_timesteps", type=int, default=14, help="")
    
    parser.add_argument("--total_train_samples", type=int, default=128, help="")    
    parser.add_argument("--global_batch_size", type=int, default=1, help="")     
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="")     
    
    # cpu loading dataset
    parser.add_argument("--num_workers", type=int, default=1, help="")
    
    # optimizer
    parser.add_argument("--optimizer", type=str, default="adamw", help="")            
    parser.add_argument("--weight_decay", type=float, default=0.01, help="")            
    parser.add_argument("--lr", type=float, default=0.0005, help="")            
    parser.add_argument("--beta1", type=float, default=0.9, help="")            
    parser.add_argument("--beta2", type=float, default=0.95, help="")            
    parser.add_argument("--eps", type=float, default=1e-08, help="")            
    
    parser.add_argument("--seed", type=int, default=42, help="")  
    parser.add_argument("--num_checkpoints", type=int, default=2, help="")  
    parser.add_argument("--dataset_manifest", type=str, default="data/droid_100_minimal/preprocessed/shards/manifest.jsonl", help="") 
    
    
    args = parser.parse_args()
    
    return args


def main():
    cfg = parse_args()

    device = 'cpu'
    # Seed rank-0 before any object creation for reproducibility.
    set_random_seed(cfg.seed, 0)

    model_cfg = load_yaml("config/vla.yaml")
    model = VLA(model_cfg)
    model = model.to(device)

    # Optionally resume model from a checkpoint or finetune from pretrained weights.
    start_checkpoint_num, global_step = 0, 0
    total_steps = cfg.total_train_samples // cfg.global_batch_size
    shard_shuffle_seed_per_dataset = None
    # Create optimizer before torchcompile
    optimizer = create_optimizer(cfg, model)
    count_parameters(model)
    # Create LR scheduler, loss function.
    scheduler = lr_scheduler
    loss = masked_mse_loss

    done_training = global_step >= total_steps
    checkpoint_num = start_checkpoint_num
    # Per-dataset cursors and shuffle seeds allow resuming mixed datasets.
    curr_shard_idx_per_dataset = 0
    if shard_shuffle_seed_per_dataset is None:
        shard_shuffle_seed_per_dataset = cfg.seed
    samples_seen = 0
    
    local_rank, global_rank, world_size = world_info_from_env()
    
    # Main training loop
    while not done_training:
        # Partition the global sample budget into evenly-sized checkpoint chunks.
        samples_per_checkpoint = cfg.total_train_samples // cfg.num_checkpoints
        datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = (
            get_datastring_input(
                num_samples=samples_per_checkpoint,
                curr_shard_idx_per_dataset=curr_shard_idx_per_dataset,
                shard_shuffle_seed_per_dataset=shard_shuffle_seed_per_dataset,
                manifest_path=cfg.dataset_manifest,
                num_workers_per_gpu=1,
                world_size=world_size,
            )
        )

        print0(f"Now training on: {datastrings}")
        print0(f"Samples: {samples_seen} / {cfg.total_train_samples}")
        print0(f"Samples in this checkpoint: {num_samples_per_dataset}")

        dataloader = get_dataloader(datastrings, num_samples_per_dataset, checkpoint_num, world_size, cfg)

        prev_step = global_step

        success, global_step = train_one_checkpoint(
            model,
            dataloader,
            loss,
            checkpoint_num,
            global_step,
            optimizer,
            scheduler,
            device,
            world_size,
            cfg,
        )

        # Translate newly completed steps into samples.
        samples_seen = samples_seen + (global_step - prev_step) * cfg.global_batch_size
        checkpoint_num += 1
        done_training = global_step >= total_steps

        if 0 == global_rank:
            # Persist training state (model/opt/scheduler + data cursors).
            checkpoint_path = "checkpoints"
            os.makedirs(checkpoint_path, exist_ok=True)
            save_checkpoint(
                checkpoint_num,
                checkpoint_path,
                model,
                optimizer,
                datastrings,
                curr_shard_idx_per_dataset,
                samples_seen,
                global_step,
                shard_shuffle_seed_per_dataset,
            )

        if done_training:
            print0("Model has seen the desired number of samples. Ending training.")
            break

def train_one_checkpoint(
    model: nn.Module,
    dataloader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
    checkpoint_num: int,
    step: int,
    optimizer: optim.Optimizer,
    scheduler: Callable[[int, int, Optimizer], None],
    device,
    world_size,
    cfg,
) -> tuple[bool, int]:
    """
    Trains model for one checkpoint on the provided data.

    This function:
      - Drives LR scheduling.
      - Performs forward/backward/step with optional gradient accumulation.
      - Computes and (optionally) all-reduces loss across ranks for logging.
      - Tracks timing/throughput metrics and logs periodically.
      - Exits when either:
          * the global training budget in samples is exhausted, or
          * the dataloader is depleted on any rank.

    Args:
        model: torch.nn.Module or a distributed-wrapped module.
        dataloader: dataloader object.
        loss_fn: Callable loss function mapping (logits, targets, mask) -> scalar loss.
        checkpoint_num: Index of the current checkpoint window (for logs).
        step: Current global training step **before** this window starts.
        optimizer: torch.optim.Optimizer instance.
        scheduler: Callable taking `step` and adjusting LR, etc.
        cfg: Training config.
        ema_model: Optional EMA model for maintaining exponential moving average of weights.

    Returns:
        success (bool): Whether training completed successfully
        step (int): Global step at the end of the checkpoint.
    """

    # Create batch handler for the model type
    batch_handler = DiffusionPolicyBatchHandler()

    model.train()

    # Let the dataloader know which window/checkpoint it's on.
    dataloader.set_checkpoint_num(checkpoint_num)
    num_batches_per_checkpoint = dataloader.dataloader.num_batches

    end = time.time()
    data_iterator = iter(dataloader.dataloader)

    # Progress bar setup - show step progress with proper starting value
    total_steps = cfg.total_train_samples // cfg.global_batch_size
    progress_bar = tqdm(
        initial=step, total=total_steps, desc=f"Checkpoint {checkpoint_num}", disable=False, unit="step"
    )

    # Open-ended loop; we break on budget or data exhaustion.
    for i in itertools.count():
        scheduler(step, total_steps, optimizer)

        # Hard-stop when we reach the sample budget translated into steps.
        if step >= total_steps:
            logging.warning(f"step: {step} has reached/exceeded total_steps: {total_steps}. ending training.")
            break

        # Try to fetch the next batch on this rank.
        try:
            batch = next(data_iterator)
            has_data = torch.tensor(1, dtype=torch.long, device=device)
        except StopIteration:
            has_data = torch.tensor(0, dtype=torch.long, device=device)


        data_time = time.time() - end
        optimizer.zero_grad()

        # Prepare model inputs and targets (including chunking) using batch handler
        model_inputs, targets, mask = batch_handler.prepare_inputs_and_targets(batch, device, cfg)

        # Validate that mask and future_mask are mutually exclusive
        if mask is not None and "future_mask" in model_inputs:
            raise ValueError(
                "mask and future_mask should not both be present. "
                "Use mask for LLM/VLM or future_mask for diffusion policy, not both."
            )

        forward_total_time = 0
        backward_total_time = 0
        total_lm_loss = 0
        assert 0 == cfg.global_batch_size % (world_size * cfg.per_gpu_batch_size)
        accum_freq = cfg.global_batch_size // (world_size * cfg.per_gpu_batch_size)
        for ii in range(accum_freq):

            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                forward_start = time.time()
                # Slice the microbatch for this accumulation step.
                start_idx = ii * cfg.per_gpu_batch_size
                end_idx = (ii + 1) * cfg.per_gpu_batch_size
                model_inputs_ii = batch_handler.slice_inputs_for_accumulation(model_inputs, start_idx, end_idx)

                if model_inputs_ii["input_ids"].shape[0] == 0:
                    break

                targets_ii = batch_handler.slice_targets_for_accumulation(
                    targets, start_idx, end_idx, sliced_inputs=model_inputs_ii
                )
                # Get mask for microbatch: use mask if present, otherwise use future_mask (diffusion policy)
                mask_ii = mask[start_idx:end_idx] if mask is not None else model_inputs_ii.get("future_mask", None)

                # Forward pass - same for all model types!
                outputs = model(**model_inputs_ii)
                forward_total_time += time.time() - forward_start

                local_loss = batch_handler.compute_loss(outputs, targets_ii, loss_fn, cfg, mask=mask_ii)

                # Scale loss by microbatch size ratio
                local_loss = local_loss * (model_inputs_ii["input_ids"].shape[0] / model_inputs["input_ids"].shape[0])

            # Backward per microbatch.
            backward_start = time.time()
            local_loss.backward()
            backward_total_time += time.time() - backward_start
            total_lm_loss += local_loss


        total_loss = total_lm_loss

        # Optimizer step
        optim_step_start = time.time()
        # (Optional) grad clipping
        # if cfg.grad_clip_norm is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, norm_type=2.0)
        optimizer.step()
        optim_step_time = time.time() - optim_step_start

        # Update timing meters for this iteration.
        batch_time = time.time() - end
        end = time.time()

        batch_count = i + 1
        step += 1  # Advance the global step after completing this batch.

        # Master-only logging & W&B
        batch_size = len(model_inputs["input_ids"])
        # update the loss meter with the global loss tensor every iteration,
        # so that the logging is of the avg of loss of the last cfg.log_every_n_steps iterations
        global_loss_tensor = total_loss.detach().clone()

        # if (
        #     (i % cfg.log_every_n_steps == 0 and i > 0)
        #     or batch_count == num_batches_per_checkpoint
        #     or step == total_steps - 1
        # ):
                # cfg=cfg,
                # batch_size=batch_size,
                # batch_num_tokens=model_inputs["input_ids"].numel(),
                # batch_count=batch_count,
                # num_batches_per_checkpoint=num_batches_per_checkpoint,
                # step=step,
                # dataloader=dataloader,
                # lr=optimizer.param_groups[0]["lr"],
                # checkpoint_num=checkpoint_num,
            

        loss_value = global_loss_tensor.item()
        # Update progress bar
        progress_bar.update(1)
        print0(f"steps:{step}/{total_steps}, loss:{loss_value:.4f}, lr:{optimizer.param_groups[0]['lr']:.6f}")

    progress_bar.close()

    return True, step

    
if __name__ == "__main__":
    main()
