import copy
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nano_robotic.data.pipeline import create_pipeline
from nano_robotic.utils.file_utils import load_dataset_manifest
from nano_robotic.data.data_utils import SharedCheckpointCounter



def seed_worker(worker_id: int) -> None:
    """
    Seed NumPy and Python RNGs inside a dataloader worker process.

    Args:
        worker_id: The worker id provided by PyTorch's DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class DataInfo:
    """
    This is a wrapper around WebDataset's DataLoader that allows us to store a few extra information.
    """

    dataloader: DataLoader
    dataset_pipelines: list[wds.DataPipeline] = None
    sampler: DistributedSampler = None
    shared_checkpoint_counter: SharedCheckpointCounter = None

    # Optional token IDs for padding and image tokenization.
    pad_token_id: int = None
    image_token_id: int = None

    def set_checkpoint_num(self, checkpoint_num: int) -> None:
        """
        Propagate the current checkpoint window number to helpers.
        """
        if self.shared_checkpoint_counter is not None:
            self.shared_checkpoint_counter.set_value(checkpoint_num)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_checkpoint_num(checkpoint_num)

    def save_configs(self, experiment_path: str):
        for dataset_pipeline in self.dataset_pipelines:
            dataset_pipeline.save_configs(experiment_path)


def get_dataloader(
    datastring: str,
    num_samples_per_dataset: int,
    checkpoint_num: int,
    world_size: int,
    cfg: object,
) -> DataInfo:
    """
    Build a mixed WebDataset dataloader for a single checkpoint window.
    Args:
        datastrings: Per-dataset WebDataset input strings.
        num_samples_per_dataset: The sample budget to draw from each dataset for this window.
            These are used as mixing probabilities in `wds.mix.RandomMix`.
        checkpoint_num: Current checkpoint window index.
        cfg: Training configuration object.

    Returns:
        DataInfo: A wrapper containing the `WebLoader` and helper objects.
    """
    shared_checkpoint_counter = SharedCheckpointCounter(checkpoint_num=checkpoint_num)

    batch_size = max(cfg.global_batch_size // world_size, 1)

    dataset = create_pipeline(datastring, checkpoint_num, batch_size)

    # Start a generator to have control over reproducibility.
    if cfg.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None


    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,  # batching handled in the pipeline
        shuffle=False,  # mixing is handled by RandomMix
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        prefetch_factor=None,
        generator=generator,
        worker_init_fn=worker_init_fn,
        in_order=False,
    )

    # Compute total batches/samples this loader will emit in this window.
    # We want each worker to process the same number of shard-groups.
    if cfg.num_workers == 0:
        logging.warning("num_workers is <= 0, setting to 1 per GPU")
    num_workers_per_gpu = max(1, cfg.num_workers)
    total_samples = num_samples_per_dataset
    denominator = cfg.global_batch_size * num_workers_per_gpu
    num_worker_batches = total_samples // denominator
    if num_worker_batches == 0:
        raise ValueError(
            f"Zero batches: total_samples ({total_samples}) < "
            f"global_batch_size ({cfg.global_batch_size}) × "
            f"num_workers ({num_workers_per_gpu}) = {denominator}. "
            f"Reduce global_batch_size or increase total_train_samples."
        )

    num_batches = num_worker_batches * num_workers_per_gpu
    num_samples = num_batches * cfg.global_batch_size

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    dataloader = DataInfo(
        dataloader=dataloader, dataset_pipelines=dataset, shared_checkpoint_counter=shared_checkpoint_counter
    )
    return dataloader


def _shuffle_manifest_inplace(manifest, seed):
    """Shuffle manifest entries in-place with a deterministic seed, matching load_dataset_manifest."""
    if seed is not None:
        np.random.default_rng(seed).shuffle(manifest)
    return manifest


def get_datastring_input(
    num_samples: int,
    curr_shard_idx_per_dataset: int,
    shard_shuffle_seed_per_dataset: int,
    manifest_path: str,
    num_workers_per_gpu: int,
    world_size: int,
) -> tuple[str|None, int, int, int]:
    """
    Select shards for the next checkpoint window and build datastrings.

    Given one or more dataset manifests, this function determines how many
    samples to draw from each dataset (according to `dataset_weighting`), then
    selects enough shards so that every worker across all ranks receives the
    same count of shards. It returns WebDataset datastrings suitable for
    `create_wds_pipeline`.

    Args:
        num_samples: Total number of samples to fetch across all
            datasets for this window.
        curr_shard_idx_per_dataset: Current shard cursor per dataset.
        shard_shuffle_seed_per_dataset: Current shuffle seed per dataset.
        manifest_paths: Paths/URIs to per-dataset manifest JSON files.
        dataset_weighting: Optional per-dataset weights; if `None`,
            uses uniform weighting.
        allow_multiple_epochs: Whether to reshuffle and wrap around when
            shards are exhausted.
        num_workers_per_gpu: Number of dataloader workers per rank.
        world_size: Total number of ranks in the distributed job.

    Returns:
        datastrings: Per-dataset WebDataset input strings (local or S3 pipe).
        num_samples_list_per_dataset: Per-dataset total samples scheduled
            for this window (after shard selection).
        next_shard_idx_per_dataset: Updated shard cursors after accounting
            for selected shards.
        next_shard_shuffle_seed_per_dataset: Updated shuffle seeds.
    """
    # Load raw manifests once (for potential in-memory re-shuffling in multi-epoch),
    # then apply the initial shuffle to get the working copies.
    raw_manifest = load_dataset_manifest(manifest_path, shard_shuffle_seed=None) 
    manifest = _shuffle_manifest_inplace(list(raw_manifest), shard_shuffle_seed_per_dataset)

    next_shard_idx_per_dataset = copy.deepcopy(curr_shard_idx_per_dataset)
    next_shard_shuffle_seed_per_dataset = copy.deepcopy(shard_shuffle_seed_per_dataset)

    # Build lists of shard names and their sample counts selected for this window.
    shard_list_per_dataset = []
    num_samples_list_per_dataset = []
    total_num_workers = num_workers_per_gpu * world_size

    # Greedily add shards until we satisfy both:
    # (a) enough samples for the weighting target, and
    # (b) at least one shard per worker (to balance work).
    needed = num_samples
    accumulated_samples = 0

    # Phase 1: Add remaining shards from current position in the current epoch.
    start_idx = curr_shard_idx_per_dataset
    for idx in range(start_idx, len(manifest)):
        shard_list_per_dataset.append(manifest[idx]["shard"])
        num_samples_list_per_dataset.append(manifest[idx]["num_sequences"])
        accumulated_samples += manifest[idx]["num_sequences"]
        curr_shard_idx_per_dataset = idx + 1
        if accumulated_samples >= needed and len(shard_list_per_dataset) >= total_num_workers:
            break


    # Ensure number of shards is a multiple of number of workers, so each worker has same number of shards.
    idx_div = (
        (len(shard_list_per_dataset) // total_num_workers) * total_num_workers
        if total_num_workers > 0
        else len(shard_list_per_dataset)
    )
    shard_list_per_dataset = shard_list_per_dataset[:idx_div]
    num_samples_list_per_dataset = num_samples_list_per_dataset[:idx_div]

    # Only add used shards. Put back unused shards.
    next_shard_idx_per_dataset += len(shard_list_per_dataset)
    if next_shard_shuffle_seed_per_dataset is not None:
        next_shard_shuffle_seed_per_dataset += next_shard_idx_per_dataset // len(manifest)
    next_shard_idx_per_dataset = next_shard_idx_per_dataset % len(manifest)

    # Build WebDataset datastrings per dataset from selected shard names.
    datastrings = None
    shard_root_source = "/".join(manifest_path.split("/")[:-1]) + "/"
    if len(shard_list_per_dataset) > 1:
        datastrings = shard_root_source + "{" + ",".join(shard_list_per_dataset) + "}.tar"
    elif len(shard_list_per_dataset) == 1:
        datastrings = shard_root_source + shard_list_per_dataset[0] + ".tar"
    else:
        logging.debug(f"No shards found for dataset in {manifest_path}")

    # Collapse per-shard sample counts into per-dataset totals.
    total_num_samples_per_dataset = sum(num_samples_list_per_dataset)

    return datastrings, total_num_samples_per_dataset, next_shard_idx_per_dataset, next_shard_shuffle_seed_per_dataset
