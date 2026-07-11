"""Base class for backbone wrappers used in action policies."""

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

class VLMBackBoneBase(nn.Module):
    """Base wrapper that adapts a vision-language model for action policy conditioning.

    Subclasses create the underlying model internally and implement
    extraction of conditioning embeddings from model outputs.
    """

    def __init__(self):
        super().__init__()
        self._model = self.create_model()

    @abstractmethod
    def get_conditioning_embeddings_dim(self) -> int:
        """Get the output conditioning embeddings dimension."""
        raise NotImplementedError
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        raise NotImplementedError

    def get_action_conditioning(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Get embeddings for conditioning action policies.

        Args:
            input_ids: Text token IDs [B, seq_len]
            pixel_values: Image tensors (format depends on model)
            attention_mask: Text attention mask
            attention_mask_images: Image attention mask
            **kwargs: Model-specific args

        Returns:
            VisionLanguageBackboneOutput with embeddings
        """
        (input_ids, pixel_values, attention_mask, attention_mask_images, model_kwargs) = self._prepare_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )

        outputs = self._model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **model_kwargs,
        )

        return self._extract_action_conditioning(model_output=outputs)

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, Any],
    ]:
        """Prepare inputs for conditioning, if necessary."""
        return input_ids, pixel_values, attention_mask, attention_mask_images, kwargs

    @abstractmethod
    def _extract_action_conditioning(self, model_output:dict) -> torch.Tensor:
        """Extract conditioning embeddings from model outputs."""
        raise NotImplementedError
