from typing import Optional, Union
import torch
from torch import nn
import math

glo_count = 0
SUM1 = 0
SUM2 = 0
temp_ratio = 2
class PLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_1_a: torch.Tensor,
        weight_1_b: torch.Tensor,
        weight_2_a: torch.Tensor,
        weight_2_b: torch.Tensor,
        average_ratio: float = 1.0,
        rank: int = 8,
        alpha: int = 1.5,
        beta: int = 0.5,
        sum_timesteps: int = 28000, 
        pattern:str = "s*",
        device: Optional[Union[torch.device, str]] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.device = device
        self.weight_1_a = weight_1_a.to(device)
        self.weight_1_b = weight_1_b.to(device)
        self.weight_2_a = weight_2_a.to(device)
        self.weight_2_b = weight_2_b.to(device)
        self.average_ratio = average_ratio
        self.sum_timesteps = sum_timesteps
        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"
        self.pattern = pattern

    def _get_dc_component(self, matrix: torch.Tensor) -> float:
        dft = torch.fft.fft2(torch.abs(matrix))
        return torch.abs(dft[0, 0]).item()

    # select topk weights 
    def get_klora_weight(self, timestep):
        global SUM1, SUM2, temp_ratio
        sum_timesteps = self.sum_timesteps
        gamma = self.average_ratio
        
        # compute the sum of top k values
        matrix1 = self.weight_1_a @ self.weight_1_b
        dc_content = self._get_dc_component(matrix1)
        SUM1 += dc_content

        matrix2 = self.weight_2_a @ self.weight_2_b
        dc_style = self._get_dc_component(matrix2)
        SUM2 += dc_style
        scale =  timestep / sum_timesteps 
        if glo_count // 4 == 0:
            return matrix1
        if glo_count % 4 == 0:
            SUM1 = SUM1 * (1 - scale) * 1.3
            SUM2 = SUM2 * gamma
            temp_ratio = SUM1 / SUM2
            SUM1 = 0
            SUM2 = 0
        
        if temp_ratio > 1:
            return matrix1
        else:
            return matrix2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        global glo_count
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1_a.dtype

        if self.forward_type == "merge":
            glo_count += 1
            weight = self.get_klora_weight(glo_count)
            if glo_count == 28000:
                glo_count = 0
        elif self.forward_type == "weight_1":
            weight = self.weight_1_a @ self.weight_1_b
        elif self.forward_type == "weight_2":
            weight = self.weight_2_a @ self.weight_2_b
        else:
            raise ValueError(self.forward_type)
        hidden_states = nn.functional.linear(hidden_states.to(dtype), weight=weight)
        return hidden_states.to(orig_dtype)


class PLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)
