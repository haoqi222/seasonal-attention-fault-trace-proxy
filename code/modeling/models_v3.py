#!/usr/bin/env python3
"""Frozen model factory for the V3 experiment protocol.

The primary deep comparison uses the same 32-channel input and a matched
capacity range. Temporal and input ablations reuse the same TemporalCMT family.
This module defines architectures only; it contains no training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from temporal_cmt import Decoder, Encoder, TemporalCMT


class PlainUNetV3(nn.Module):
    """Four-level convolutional U-Net with a 32-channel stacked input."""

    def __init__(self, in_channels: int = 32, base: int = 16, dropout: float = 0.0):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, base=base, dropout=dropout)
        self.decoder = Decoder(base=base, out_ch=1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(*features[::-1])


DEEP_MODEL_IDS = (
    "unet_matched",
    "temporal_attn",
    "temporal_concat",
    "summer_only",
    "thermal_only",
    "optical_only",
)


def build_deep_model(model_id: str) -> nn.Module:
    """Build one protocol-frozen deep model without moving it to a device."""

    if model_id == "unet_matched":
        return PlainUNetV3(in_channels=32, base=16, dropout=0.0)
    if model_id == "temporal_attn":
        return TemporalCMT(
            season_sel=[0, 1, 2, 3], temporal="attn", base=16, heads=4,
            dropout=0.0, band_sel=list(range(8)), n_bands=8,
        )
    if model_id == "temporal_concat":
        return TemporalCMT(
            season_sel=[0, 1, 2, 3], temporal="concat", base=16, heads=4,
            dropout=0.0, band_sel=list(range(8)), n_bands=8,
        )
    if model_id == "summer_only":
        return TemporalCMT(
            season_sel=[1], temporal="attn", base=16, heads=4,
            dropout=0.0, band_sel=list(range(8)), n_bands=8,
        )
    if model_id == "thermal_only":
        return TemporalCMT(
            season_sel=[0, 1, 2, 3], temporal="attn", base=16, heads=4,
            dropout=0.0, band_sel=[0], n_bands=8,
        )
    if model_id == "optical_only":
        return TemporalCMT(
            season_sel=[0, 1, 2, 3], temporal="attn", base=16, heads=4,
            dropout=0.0, band_sel=list(range(1, 8)), n_bands=8,
        )
    raise ValueError(f"Unknown V3 deep model {model_id!r}; choose from {DEEP_MODEL_IDS}")


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
