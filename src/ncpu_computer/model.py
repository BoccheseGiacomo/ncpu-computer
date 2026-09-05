from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def perception_kernel(name: str, random_seed: int = 0) -> torch.Tensor:
    if name == "identity":
        kernel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    elif name == "sobel_x":
        kernel = torch.tensor(
            [[-0.25, 0.0, 0.25], [-0.5, 0.0, 0.5], [-0.25, 0.0, 0.25]]
        )
    elif name == "sobel_y":
        kernel = torch.tensor(
            [[-0.25, -0.5, -0.25], [0.0, 0.0, 0.0], [0.25, 0.5, 0.25]]
        )
    elif name == "laplacian":
        kernel = torch.tensor([[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]])
    elif name == "random":
        generator = torch.Generator().manual_seed(random_seed)
        kernel = torch.randn(3, 3, generator=generator)
        kernel = kernel / kernel.norm()
    else:
        raise ValueError(f"unknown perception kernel: {name}")
    return kernel


class Perception(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.channels = config.channels
        self.padding = config.padding
        fixed_names = list(config.fixed_kernels)
        if config.fixed_laplacian:
            fixed_names.append("laplacian")
        self.fixed_names = tuple(fixed_names)
        self.learnable_count = config.learnable_kernels
        for index, name in enumerate(self.fixed_names):
            self.register_buffer(f"fixed_{index}", perception_kernel(name))
        for index in range(self.learnable_count):
            kernel = perception_kernel(
                config.learnable_kernel_init,
                config.random_kernel_seed + index,
            )
            self.register_parameter(f"learnable_{index}", nn.Parameter(kernel))

    @property
    def kernel_count(self) -> int:
        return len(self.fixed_names) + self.learnable_count

    def kernel_bank(self) -> torch.Tensor:
        kernels = [
            getattr(self, f"fixed_{index}") for index in range(len(self.fixed_names))
        ]
        kernels.extend(
            getattr(self, f"learnable_{index}") for index in range(self.learnable_count)
        )
        return torch.stack(kernels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        kernels = self.kernel_bank()
        filters = (
            kernels.unsqueeze(0)
            .expand(self.channels, -1, -1, -1)
            .reshape(self.channels * self.kernel_count, 1, 3, 3)
        )
        padding_mode = "constant" if self.padding == "zeros" else self.padding
        padded = F.pad(state, (1, 1, 1, 1), mode=padding_mode)
        return F.conv2d(padded, filters, groups=self.channels)


class UpdateRule(nn.Module):
    def __init__(self, config: ModelConfig, perception_channels: int):
        super().__init__()
        self.channels = config.channels
        self.gate = config.gate
        output_channels = (
            config.channels if config.gate == "none" else 2 * config.channels
        )
        self.hidden = nn.Conv2d(perception_channels, config.hidden_size, 1)
        self.output = nn.Conv2d(
            config.hidden_size,
            output_channels,
            1,
            bias=config.gate != "none",
        )
        nn.init.zeros_(self.output.weight[: config.channels])
        if config.gate != "none":
            nn.init.zeros_(self.output.bias[: config.channels])
            nn.init.zeros_(self.output.weight[config.channels :])
            nn.init.constant_(self.output.bias[config.channels :], config.gate_bias)

    def forward(self, perception: torch.Tensor) -> torch.Tensor:
        output = self.output(F.relu(self.hidden(perception)))
        if self.gate == "none":
            return output
        delta, gate_values = output.split(self.channels, dim=1)
        if self.gate == "linear":
            gate = gate_values
        elif self.gate == "sigmoid":
            gate = torch.sigmoid(gate_values)
        elif self.gate == "tanh":
            gate = torch.tanh(gate_values)
        elif self.gate == "relu":
            gate = F.relu(gate_values)
        else:
            raise RuntimeError(f"invalid gate configured: {self.gate}")
        return delta * gate


class NeuralCellularAutomaton(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.perception = Perception(config)
        self.rule = UpdateRule(config, config.channels * self.perception.kernel_count)
        update_mask = torch.ones(1, config.channels, 1, 1)
        update_mask[:, config.program_channel] = 0.0
        self.register_buffer("update_mask", update_mask)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(self, io_grid: torch.Tensor) -> torch.Tensor:
        if io_grid.ndim != 3:
            raise ValueError("io_grid must have shape (batch, height, width)")
        if not torch.is_floating_point(io_grid):
            raise ValueError("io_grid must be floating point")
        state = torch.zeros(
            io_grid.shape[0],
            self.config.channels,
            io_grid.shape[1],
            io_grid.shape[2],
            device=io_grid.device,
            dtype=io_grid.dtype,
        )
        state[:, self.config.io_channel] = io_grid
        return state

    def step(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 4 or state.shape[1] != self.config.channels:
            raise ValueError(
                f"state must have shape (batch, {self.config.channels}, height, width)"
            )
        delta = self.rule(self.perception(state)) * self.update_mask
        if self.config.fire_rate < 1.0:
            fire_mask = (
                torch.rand(
                    state.shape[0],
                    1,
                    state.shape[2],
                    state.shape[3],
                    device=state.device,
                )
                < self.config.fire_rate
            )
            delta = delta * fire_mask
        updated = state + delta
        if self.config.max_abs_state is not None:
            updated = updated.clamp(
                -self.config.max_abs_state, self.config.max_abs_state
            )
        updated = torch.where(self.update_mask.bool(), updated, state)
        return updated

    def forward(self, initial_state: torch.Tensor, steps: int) -> torch.Tensor:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        states = [initial_state]
        state = initial_state
        for _ in range(steps):
            state = self.step(state)
            states.append(state)
        return torch.stack(states, dim=1)
