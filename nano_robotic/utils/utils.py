import random
import re
import warnings
from collections import Counter, defaultdict
from typing import Any
import yaml
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import os

def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def print0(s: str, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    if 0 == rank:
        print(s, **kwargs)

def freeze_parameters(module : nn.Module):
    """Freeze all parameters in the model by setting requires_grad to False."""
    for param in module.parameters():
        param.requires_grad = False

def maybe_get_current_commit_sha(default: str | None = None) -> str | None:
    """Return the current HEAD commit SHA, or ``default`` if not in a git repo."""
    try:
        import git

        return git.Repo(search_parent_directories=True).head.object.hexsha
    except Exception:
        warnings.warn("Could not determine git commit SHA.", stacklevel=2)
        return default


def maybe_get_remote_url_from_active_branch(default: str | None = None) -> str | None:
    """Return the remote URL of the active branch's tracked remote, or ``default``."""
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        tracking = repo.active_branch.tracking_branch()
        if tracking and tracking.remote_name:
            for remote in repo.remotes:
                if remote.name == tracking.remote_name:
                    return remote.url
        return default
    except Exception:
        warnings.warn("Could not determine git remote URL.", stacklevel=2)
        return default


def set_random_seed(seed: int = 42, rank: int = 0) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Base seed.
        rank: Rank-specific offset to decorrelate RNG streams across processes.
    """
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def load_yaml(file: str) -> Any:
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

# no recursive conversion for now
def to_dict(obj: SimpleNamespace) -> dict[str, Any]:
    # for name, value in obj.__item__:
    return vars(obj)
        

def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    else:
        return obj

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")    

def summarize_datastrings(datastrings: list[str]) -> str:
    """
    Sometimes datastrings can be very long (e.g., many epochs). This helper function
    summarize them to avoid polluting logging.
    """
    datastring_pattern = re.compile(r"^(?P<prefix>.*?){(?P<items>[^}]*)}(?P<suffix>.*)$")

    # (prefix, suffix) -> Counter[str, count]
    counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    # Data strings without braces
    passthrough = Counter()

    for ds in datastrings:
        match = datastring_pattern.match(ds)
        if match:
            prefix, items, suffix = match.group("prefix", "items", "suffix")
            key = (prefix, suffix)
            vals = [item.strip() for item in items.split(",")]
            for val in vals:
                counter[key][val] += 1
        else:
            passthrough[ds] += 1

    parts = []
    for (prefix, suffix), item_counter in sorted(counter.items()):
        items = sorted(item_counter.keys())
        counts = [item_counter[item] for item in items]
        uniform = all(c == counts[0] for c in counts)
        if uniform:
            part = f"{prefix}{{{', '.join(items)}}}{suffix}"
            if counts[0] > 1:
                part += f" (x{counts[0]})"
        else:
            item_parts = [f"{item} (x{item_counter[item]})" for item in items]
            part = f"{prefix}{{{', '.join(item_parts)}}}{suffix}"
        parts.append(part)

    for ds, count in sorted(passthrough.items()):
        part = ds
        if count > 1:
            part += f" (x{count})"
        parts.append(part)

    return "[" + ",".join(parts) + "]"
