import numpy as np
from torch.optim.optimizer import Optimizer

def lr_scheduler(step, total_steps, optimizer: Optimizer):
    
    base_lr         = 0.0005
    warmup_length   = 2
    force_min_lr    = 0.0
    min_lr          = 1e-05
    if step < warmup_length:
        lr = base_lr * (step + 1) / warmup_length
    else:
        e = step - warmup_length
        es = total_steps - warmup_length
        lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr)
        lr = max(lr, force_min_lr)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
    return lr