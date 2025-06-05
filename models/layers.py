import math
import torch
from torch import nn


class LatentModulatedSIRENLayer(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        condition_dim: 6,
        condition_hidden_dim: 256,
        latent_modulation_dim: 512,
        conditioning_type="concatenation",
        w0=30.0,
        modulate_shift=True,
        modulate_scale=False,
        is_first=False,
        is_last=False,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.conditioning_type = conditioning_type
        self.is_last = is_last

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
            if self.conditioning_type == "concatenation":
                self.condition_shift_layer = nn.Linear(
                    latent_modulation_dim, out_size // 2
                )
                self.modulate_shift_layer = nn.Linear(
                    latent_modulation_dim, out_size // 2
                )
            elif self.conditioning_type == "mlp":
                self.condition_shift_layer = nn.Linear(latent_modulation_dim, out_size)
        if modulate_scale:
            if self.conditioning_type == "concatenation":
                self.condition_scale_layer = nn.Linear(
                    latent_modulation_dim, out_size // 2
                )
                self.modulate_scale_layer = nn.Linear(
                    latent_modulation_dim, out_size // 2
                )
            elif self.conditioning_type == "mlp":
                self.condition_scale_layer = nn.Linear(latent_modulation_dim, out_size)

        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1 / dim_in if is_first else (math.sqrt(6.0 / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x, latent, factors=None):
        x = self.linear(x)
        if not self.is_last:
            if not self.modulate_shift:
                shift = 0.0
            else:
                if self.conditioning_type == "concatenation":
                    # shift_condition = self.modulate_shift_layer(factors)
                    shift_condition = self.condition_shift_layer(factors)
                    shift_modulation = self.modulate_shift_layer(latent)
#                     print(shift_condition.shape, shift_modulation.shape)
                    shift = torch.cat([shift_condition, shift_modulation], dim=-1)
                elif self.conditioning_type == "mlp":
                    shift = self.condition_shift_layer(factors)
            if not self.modulate_scale:
                scale = 1.0
            else:
                if self.conditioning_type == "concatenation":
                    scale_condition = self.condition_scale_layer(factors)
                    scale_modulation = self.modulate_scale_layer(latent)
                    scale = torch.cat([scale_condition, scale_modulation], dim=-1)
                elif self.conditioning_type == "mlp":
                    scale = self.condition_scale_layer(factors)

            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)

            x = scale * x + shift
            x = torch.sin(self.w0 * x)
        return x
