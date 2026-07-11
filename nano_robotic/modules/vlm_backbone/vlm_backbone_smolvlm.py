"""VLM backbone wrapper for action policy conditioning."""

import torch

from nano_robotic.modules.vlm_backbone.vlm_backbone_base import VLMBackBoneBase
from nano_robotic.utils.utils import freeze_parameters
from nano_robotic.utils.model_utils import get_hidden_dim_hf, get_num_hidden_layers_hf
from transformers import AutoConfig, AutoModelForImageTextToText


class VLMHFBackboneWrapper(VLMBackBoneBase):
    """VLM backbone wrapper for action policy conditioning.

    This class wraps VLMHF and provides the conditioning embedding interface
    using an action token probe, while keeping VLMHF free from conditioning-specific logic.
    """

    def __init__(self):
        """Initialize VLM backbone for conditioning.

        Args:
            backbone_params: Configuration parameters for the VLM backbone.
            load_pretrained: If True, download pretrained weights.
        """
        super().__init__()
                
        self.num_vlm_layers_to_use = 1 # suppose is 1

        # Add action token to vocabulary
        self._action_token_id = self._model.resize_token_embeddings()

        # Detect model type
        self._vlm_model_type = getattr(self._model.model.config, "model_type", "").lower()

        # Initialize the new token's embedding as mean of existing embeddings
        with torch.no_grad():
            embeddings = self._model.model.get_input_embeddings()
            mean_emb = embeddings.weight[:-1].mean(dim=0)
            embeddings.weight[-1] = mean_emb

    def create_model(self) -> torch.nn.Module:
        return VLMHF()

    def get_conditioning_embeddings_dim(self) -> int:
        """Return dimension for condition embeddings (with layers concatenated)."""
        return self._model.lm_hidden_dim * self.num_vlm_layers_to_use

    @property
    def _is_qwen_style(self) -> bool:
        return "qwen" in self._vlm_model_type

    @property
    def _is_paligemma_style(self) -> bool:
        return "paligemma" in self._vlm_model_type

    def _validate_inputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Validate VLM backbone inputs.

        Ensures required arguments are present and consistent (e.g., Qwen image_grid_thw
        matches pixel_values patches). Raises ValueError if inputs are invalid.
        """
        if self._is_qwen_style and "image_grid_thw" not in kwargs:
            raise ValueError(
                f"Qwen model requires image_grid_thw but it was not provided. "
                f"pixel_values shape: {pixel_values.shape}. "
                f"Ensure the processor passes image_grid_thw through the data pipeline."
            )

    def _prepare_inputs_for_action_token(self, input_ids, attention_mask):
        """Helper: Append action token to inputs (modular and reusable)."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Append action token
        action_tokens = torch.full((batch_size, 1), self._action_token_id, device=device, dtype=input_ids.dtype)
        input_ids_with_action = torch.cat([input_ids, action_tokens], dim=1)

        # Extend attention mask
        if attention_mask is not None:
            action_mask = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            attention_mask_with_action = torch.cat([attention_mask, action_mask], dim=1)
        else:
            attention_mask_with_action = None

        return input_ids_with_action, attention_mask_with_action

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ):
        self._validate_inputs(input_ids, pixel_values, attention_mask, attention_mask_images, **kwargs)

        input_ids_with_action, attention_mask_with_action = self._prepare_inputs_for_action_token(
            input_ids, attention_mask
        )

        # VLM needs hidden states for action token embedding extraction
        kwargs["output_hidden_states"] = True

        return input_ids_with_action, pixel_values, attention_mask_with_action, attention_mask_images, kwargs

    def _extract_action_conditioning(self, model_output) -> torch.Tensor:
        embeddings = self._extract_action_token_embeddings(model_output)
        return embeddings

    def _extract_action_token_embeddings(self, outputs):
        """Helper: Extract and concatenate hidden states from action token position."""
        hidden_states = outputs.hidden_states
        action_hidden_states = []
        for layer_hidden in hidden_states[-self.num_vlm_layers_to_use :]:
            action_hidden = layer_hidden[:, -1, :]  # [B, hidden_dim]
            action_hidden_states.append(action_hidden)

        # Concatenate: [B, 1, hidden_dim * num_vlm_layers_to_use]
        return torch.cat(action_hidden_states, dim=-1).unsqueeze(1)



class VLMHF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
            # self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)
        self._limit_hidden_states_to_last_n = None
        self._setup_model_info()

    def _setup_model_info(self):
        """Detect model-specific information like hidden dimensions and expected image size."""
        model = self.model
        config = model.config

        # Detect language model component
        language_model_ref = None
        for attr in ["language_model", "text_model", "llm", "model"]:
            if hasattr(model, attr):
                candidate = getattr(model, attr)
                if candidate is not model and hasattr(candidate, "forward"):
                    language_model_ref = candidate
                    break

        # Detect language model hidden dimension
        self._lm_hidden_dim = None
        if language_model_ref is not None:
            lm_config = getattr(language_model_ref, "config", None)
            if lm_config is not None:
                for attr in ["hidden_size", "d_model", "n_embd", "hidden_dim"]:
                    if hasattr(lm_config, attr):
                        self._lm_hidden_dim = getattr(lm_config, attr)
                        break

        # Detect patches per image from vision config
        self._patches_per_image = None
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            img_size = getattr(vision_config, "image_size", None)
            patch_size = getattr(vision_config, "patch_size", 14)
            if isinstance(img_size, int) and isinstance(patch_size, int):
                self._patches_per_image = (img_size // patch_size) ** 2
            elif isinstance(img_size, (list, tuple)) and isinstance(patch_size, int):
                self._patches_per_image = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # SmolVLM/Idefics-family models expect pixel_values as 5D [B, N, C, H, W]
        # (per-sample image count in the second dim). PaliGemma/CLIP/Qwen-VL do not.
        model_type = getattr(config, "model_type", "").lower()
        self._expects_5d_pixel_values = model_type in {"smolvlm", "idefics2", "idefics3"}

    @property
    def lm_hidden_dim(self):
        """Get the language model's hidden dimension (may differ from embedding dim)."""
        if self._lm_hidden_dim is not None:
            return self._lm_hidden_dim
        # Fallback to general hidden_dim
        return self.hidden_dim

    def forward(self, input_ids, pixel_values, attention_mask=None, output_hidden_states=False, **kwargs):
        # SmolVLM/Idefics expect 5D [B, N, C, H, W]; reshape flat 4D from
        # processors like simple_vlm using batch_size from input_ids.
        if (
            pixel_values is not None
            and pixel_values.ndim == 4
            and self._expects_5d_pixel_values
            and input_ids is not None
        ):
            batch_size = input_ids.shape[0]
            num_images = pixel_values.shape[0] // batch_size
            if num_images * batch_size != pixel_values.shape[0]:
                raise ValueError(
                    f"pixel_values batch size ({pixel_values.shape[0]}) is not divisible by "
                    f"input_ids batch size ({batch_size}); cannot infer images-per-sample."
                )
            pixel_values = pixel_values.reshape(batch_size, num_images, *pixel_values.shape[1:])

        out = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        if self._limit_hidden_states_to_last_n is not None and output_hidden_states:
            out.hidden_states = out.hidden_states[-self._limit_hidden_states_to_last_n :]

        return out

    def resize_token_embeddings(self, new_num_tokens: int = None) -> int:
        """Add a new token to the vocabulary and return its ID.

        Args:
            new_num_tokens: The new vocabulary size. If None, adds exactly one token.

        Returns:
            The ID (index) of the newly added token.
        """
        current_size = int(self.model.get_input_embeddings().num_embeddings)

        if new_num_tokens is None:
            new_num_tokens = current_size + 1

        if new_num_tokens > current_size:
            print(f"Resizing token embeddings from {current_size} to {new_num_tokens}")
            self.model.resize_token_embeddings(new_num_tokens, mean_resizing=False)

        # Return the ID of the last token (the newly added one)
        return new_num_tokens - 1

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            if enable:
                self.model.gradient_checkpointing_enable()
            else:
                self.model.gradient_checkpointing_disable()

    @property
    def hidden_dim(self) -> int:
        return get_hidden_dim_hf(self.model.config)

    @property
    def num_hidden_layers(self) -> int:
        return get_num_hidden_layers_hf(self.model.config)

    def generate(self, input_ids, pixel_values, attention_mask, max_new_tokens=20, **kwargs):
        """Generate text tokens using the VLM HF model"""
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        generated = input_ids.clone()
        attn_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids=generated, pixel_values=pixel_values, attention_mask=attn_mask, **kwargs)
            last_output = outputs.logits[:, -1, :]
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask: 1 for non-padding tokens
            next_token_mask = torch.ones_like(next_token, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, next_token_mask], dim=-1)

        return generated

    def get_fsdp_block_types(self):
        """Return block types for FSDP wrapping."""
        block_types = set()

        # Find text/language model blocks
        for attr in ["language_model", "text_model"]:
            if hasattr(self.model.model, attr):
                for _name, module in getattr(self.model.model, attr).named_modules():
                    if isinstance(module, nn.ModuleList) and len(module) > 0:
                        block_types.add(type(module[0]))

        # Find vision model blocks
        if hasattr(self.model.model, "vision_model") and hasattr(self.model.model.vision_model, "encoder"):
            for _name, module in self.model.model.vision_model.encoder.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    block_types.add(type(module[0]))

        if not block_types:
            raise ValueError("Could not find any model block classes.")

        return tuple(block_types)