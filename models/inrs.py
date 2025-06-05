import math
import torch
from torch import nn

from models.layers import LatentModulatedSIRENLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        w0=30.0,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=1024,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.w0 = w0

        layers = []
        if nlayers == 1:
            fc = nn.Linear(in_dim, bottleneck_dim)
            self._init(fc, is_first=True)
            layers.append(fc)
        else:
            first_fc = nn.Linear(in_dim, hidden_dim)
            self._init(first_fc, is_first=True)
            layers.append(first_fc)
            if use_bn:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            for _ in range(nlayers - 2):
                fc = nn.Linear(hidden_dim, bottleneck_dim)
                self._init(fc, is_first=False)
                layers.append(fc)
                if use_bn:
                    layers.append(nn.LayerNorm(bottleneck_dim))
                layers.append(nn.SiLU())

            self.mlp = nn.Sequential(*layers)

            self.final_fc = nn.Linear(bottleneck_dim, out_dim)
            self._init(self.final_fc, is_first=False)

    def _init(self, layer, is_first):
        if isinstance(layer, nn.Linear):
            dim_in = layer.weight.size(1)
            w_std = 1 / dim_in if is_first else (math.sqrt(6.0 / dim_in) / self.w0)
            nn.init.uniform_(layer.weight, -w_std, w_std)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -w_std, w_std)

    def forward(self, x, return_modulation=False):
        modulation = self.mlp(x)
        x = self.final_fc(modulation)
        if return_modulation:
            return x, modulation
        else:
            return x


class LatentModulatedSIREN(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        condition_dim=6,
        condition_hidden_dim=512,
        hidden_size=256,
        conditioning_type="concatenation",
        num_layers=5,
        latent_modulation_dim=512,
        w0=30.0,
        w0_increments=0.0,
        modulate_shift=True,
        modulate_scale=False,
        enable_skip_connections=True,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            layers.append(
                LatentModulatedSIRENLayer(
                    in_size=layer_in_size,
                    out_size=hidden_size,
                    condition_dim=condition_dim,
                    condition_hidden_dim=condition_hidden_dim,
                    conditioning_type=conditioning_type,
                    latent_modulation_dim=latent_modulation_dim,
                    w0=w0,
                    modulate_shift=modulate_shift,
                    modulate_scale=modulate_scale,
                    is_first=is_first,
                )
            )
            w0 += w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer(
            in_size=hidden_size,
            out_size=out_size,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
            conditioning_type=conditioning_type,
            latent_modulation_dim=latent_modulation_dim,
            w0=w0,
            modulate_shift=modulate_shift,
            modulate_scale=modulate_scale,
            is_last=True,
        )
        self.mlp = ConditionMLP(
            in_dim=condition_dim,
            out_dim=latent_modulation_dim,
            hidden_dim=condition_hidden_dim * 2,
            bottleneck_dim=condition_hidden_dim,
        )
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(
            size=[latent_modulation_dim], requires_grad=True
        ).to(device)

    def reset_modulations(self):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True

    def forward(self, x, conditions, get_features=False):
        #         print("x: ", x.shape)
        factors = self.mlp(conditions)
        x = self.layers[0](x, self.modulations, factors)
        for layer in self.layers[1:]:
            y = layer(x, self.modulations, factors)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, self.modulations, factors) + 0.5

        if get_features:
            return out, features
        else:
            return out
