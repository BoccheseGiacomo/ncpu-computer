from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import GeometryConfig, ModelConfig
from .tape import TapeLayout


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
    def __init__(
        self,
        config: ModelConfig,
        geometry: GeometryConfig,
        task_names: tuple[str, ...],
    ):
        super().__init__()
        config.validate()
        geometry.validate()
        if not task_names or any(
            not isinstance(name, str) or not name.strip() for name in task_names
        ):
            raise ValueError("task names must be non-empty strings")
        if len(set(task_names)) != len(task_names):
            raise ValueError("task names must be unique")
        self.config = config
        self.geometry = geometry
        self.task_names = tuple(task_names)
        self.layout = TapeLayout(geometry)
        program_shape = (
            len(task_names),
            config.program_channels,
            self.layout.height,
            self.layout.width,
        )
        if config.program_placement == "tape":
            program_shape = (
                len(task_names),
                config.program_channels,
                geometry.tape_slots,
            )
        self.programs = nn.Parameter(torch.empty(program_shape))
        nn.init.normal_(self.programs, std=config.program_init_std)
        self.perception = Perception(config)
        self.rule = UpdateRule(config, config.channels * self.perception.kernel_count)
        update_mask = torch.ones(1, config.channels, 1, 1)
        if not config.program_mutable:
            update_mask[:, : config.program_channels] = 0.0
        if config.io_mode == "separate" and config.input_mode == "frozen":
            update_mask[:, config.input_channel] = 0.0
        self.register_buffer("update_mask", update_mask)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def shared_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for name, parameter in self.named_parameters()
            if name != "programs"
        )

    def task_index(self, task_name: str) -> int:
        try:
            return self.task_names.index(task_name)
        except ValueError as error:
            raise ValueError(f"unknown task: {task_name!r}") from error

    def program_grid(self, task_indices: torch.Tensor) -> torch.Tensor:
        task_indices = torch.as_tensor(
            task_indices, dtype=torch.int64, device=self.programs.device
        )
        if task_indices.ndim != 1:
            raise ValueError("task_indices must be one-dimensional")
        if bool(((task_indices < 0) | (task_indices >= len(self.task_names))).any()):
            raise ValueError("task index is out of range")
        selected = self.programs[task_indices]
        if self.config.program_placement == "tape":
            selected = self.layout.render_tape(selected)
        return selected

    def initial_state(
        self, input_grid: torch.Tensor, task_indices: torch.Tensor
    ) -> torch.Tensor:
        if input_grid.ndim != 3:
            raise ValueError("input_grid must have shape (batch, height, width)")
        if not torch.is_floating_point(input_grid):
            raise ValueError("input_grid must be floating point")
        if input_grid.shape[1:] != (self.layout.height, self.layout.width):
            raise ValueError("input grid does not match the fixed model geometry")
        if input_grid.device != self.device:
            raise ValueError("input grid and model must be on the same device")
        task_indices = torch.as_tensor(
            task_indices, dtype=torch.int64, device=self.programs.device
        )
        if task_indices.ndim != 1 or task_indices.shape[0] != input_grid.shape[0]:
            raise ValueError("one task index is required per input")
        state = torch.zeros(
            input_grid.shape[0],
            self.config.channels,
            input_grid.shape[1],
            input_grid.shape[2],
            device=input_grid.device,
            dtype=input_grid.dtype,
        )
        state[:, : self.config.program_channels] = self.program_grid(task_indices).to(
            dtype=input_grid.dtype
        )
        state[:, self.config.input_channel] = input_grid
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
        return torch.where(self.update_mask.bool(), updated, state)

    def forward(self, initial_state: torch.Tensor, steps: int) -> torch.Tensor:
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        states = [initial_state]
        state = initial_state
        for _ in range(steps):
            state = self.step(state)
            states.append(state)
        return torch.stack(states, dim=1)
