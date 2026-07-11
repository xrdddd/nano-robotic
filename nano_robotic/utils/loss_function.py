import torch
from torch import Tensor

def masked_mse_loss(input: Tensor, target: Tensor, mask=None) -> Tensor:
    """
    Compute the MSE loss with masking. If no mask is provided, all values are considered valid.

    Args:
        input: Predicted values
        target: Ground truth values
        mask: Mask of valid actions. Either same shape as input, or same shape as input without last dimension.

    Returns:
        Masked MSE loss between inputs and targets
    """
    if mask is None:
        mask = torch.ones_like(input)
    elif mask.shape == input.shape:
        pass
    elif mask.shape == input.shape[:-1]:
        mask = mask.unsqueeze(-1).expand_as(input)
    else:
        raise ValueError(f"Mask shape {mask.shape} is not compatible with input shape {input.shape}")

    if mask.sum() == 0:
        return torch.tensor(0.0)
    else:
        return torch.nn.functional.mse_loss(input, target, weight=mask)