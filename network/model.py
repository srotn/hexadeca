"""Configurable residual policy-value network for Hexadeca."""

from __future__ import annotations

from typing import NamedTuple, cast

import torch
from torch import nn

from network.errors import NetworkInputError
from network.specification import NetworkSpecification


class NetworkOutput(NamedTuple):
    """Batched policy/win logits with normalized score predictions."""

    policy_logits: torch.Tensor
    black_score: torch.Tensor
    white_score: torch.Tensor
    win_logit: torch.Tensor

    @property
    def win_probability(self) -> torch.Tensor:
        """Return the current player's numerically stable win probability."""

        return torch.sigmoid(self.win_logit)


class ResidualBlock(nn.Module):
    """Two-convolution residual block preserving shape and channel count."""

    def __init__(self, specification: NetworkSpecification) -> None:
        super().__init__()
        channels = specification.channels
        self.convolution_1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.batch_norm_1 = nn.BatchNorm2d(
            channels,
            eps=specification.batch_norm_epsilon,
            momentum=specification.batch_norm_momentum,
        )
        self.convolution_2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.batch_norm_2 = nn.BatchNorm2d(
            channels,
            eps=specification.batch_norm_epsilon,
            momentum=specification.batch_norm_momentum,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the residual transform and post-addition ReLU."""

        residual = inputs
        outputs = self.convolution_1(inputs)
        outputs = self.batch_norm_1(outputs)
        outputs = self.activation(outputs)
        outputs = self.convolution_2(outputs)
        outputs = self.batch_norm_2(outputs)
        return cast(torch.Tensor, self.activation(outputs + residual))


class PolicyValueNetwork(nn.Module):
    """Shared ResNet trunk with policy, score, and win predictions."""

    def __init__(self, specification: NetworkSpecification) -> None:
        super().__init__()
        specification.validate()
        self.specification = specification
        self.stem = nn.Sequential(
            nn.Conv2d(
                specification.input_planes,
                specification.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                specification.channels,
                eps=specification.batch_norm_epsilon,
                momentum=specification.batch_norm_momentum,
            ),
            nn.ReLU(inplace=True),
        )
        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(specification)
                for _ in range(specification.residual_blocks)
            ]
        )

        self.policy_convolution = nn.Conv2d(
            specification.channels,
            specification.policy_head_channels,
            kernel_size=1,
            bias=False,
        )
        self.policy_batch_norm = nn.BatchNorm2d(
            specification.policy_head_channels,
            eps=specification.batch_norm_epsilon,
            momentum=specification.batch_norm_momentum,
        )
        self.policy_activation = nn.ReLU(inplace=True)
        policy_features = (
            specification.policy_head_channels
            * specification.board_size
            * specification.board_size
        )
        self.policy_linear = nn.Linear(policy_features, specification.policy_size)

        self.value_convolution = nn.Conv2d(
            specification.channels,
            specification.value_head_channels,
            kernel_size=1,
            bias=False,
        )
        self.value_batch_norm = nn.BatchNorm2d(
            specification.value_head_channels,
            eps=specification.batch_norm_epsilon,
            momentum=specification.batch_norm_momentum,
        )
        self.value_activation = nn.ReLU(inplace=True)
        value_features = (
            specification.value_head_channels
            * specification.board_size
            * specification.board_size
        )
        self.value_hidden = nn.Linear(
            value_features, specification.value_hidden_features
        )
        self.value_output = nn.Linear(specification.value_hidden_features, 3)
        self._initialize_parameters()

    def forward(self, inputs: torch.Tensor) -> NetworkOutput:
        """Return policy logits and value predictions for an NCHW batch."""

        expected_shape = (
            self.specification.input_planes,
            self.specification.board_size,
            self.specification.board_size,
        )
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected_shape:
            raise NetworkInputError(
                "Expected input shape [N, "
                f"{expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}]"
            )
        if inputs.shape[0] == 0:
            raise NetworkInputError("Network input batch must not be empty")
        if not inputs.is_floating_point():
            raise NetworkInputError("Network inputs must use a floating dtype")

        trunk = self.residual_tower(self.stem(inputs))

        policy = self.policy_convolution(trunk)
        policy = self.policy_batch_norm(policy)
        policy = self.policy_activation(policy)
        policy_logits = self.policy_linear(torch.flatten(policy, start_dim=1))

        value = self.value_convolution(trunk)
        value = self.value_batch_norm(value)
        value = self.value_activation(value)
        value = self.value_activation(
            self.value_hidden(torch.flatten(value, start_dim=1))
        )
        black_logit, white_logit, win_logit = self.value_output(value).unbind(dim=1)
        return NetworkOutput(
            policy_logits=policy_logits,
            black_score=torch.sigmoid(black_logit),
            white_score=torch.sigmoid(white_logit),
            win_logit=win_logit,
        )

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
