# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Constant-velocity trajectory model for ground-truth replay."""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..schema import ModelConfig
from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction, PredictionInput

logger = logging.getLogger(__name__)


class StaticModel(BaseTrajectoryModel):
    """Extends the current speed straight ahead, with no network and no GPU.

    Rendered camera frames reach the ASL only as ``driver_camera_image``
    entries, so a rollout that replays the logged trajectory for its whole
    duration still has to query a driver for the observations to be recorded.
    This model is that query's cheapest possible answer: it ignores the images
    and returns an arc-free trajectory at the observed speed, which force-GT
    discards.
    """

    TRAJECTORY_HORIZON_S = 4.0

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "StaticModel":
        """Create StaticModel from driver configuration."""
        return cls(
            camera_ids=camera_ids,
            output_frequency_hz=output_frequency_hz,
            context_length=context_length or 1,
        )

    def __init__(
        self,
        camera_ids: list[str],
        output_frequency_hz: int,
        context_length: int = 1,
    ):
        """Initialize StaticModel.

        Args:
            camera_ids: Camera logical IDs the servicer will deliver. The images
                are not read, but the set must match what the driver is
                configured to request.
            output_frequency_hz: Trajectory output frequency in Hz.
            context_length: Number of temporal frames to buffer. One frame is
                enough; a longer context only delays the first prediction.
        """
        if not camera_ids:
            raise ValueError("StaticModel requires at least 1 camera")

        self._camera_ids = camera_ids
        self._context_length = context_length
        self._output_frequency_hz = output_frequency_hz
        self._num_waypoints = int(self.TRAJECTORY_HORIZON_S * output_frequency_hz)

        logger.info(
            "Initialized StaticModel with %d camera(s), output_frequency=%dHz, "
            "horizon=%.1fs (%d waypoints)",
            len(camera_ids),
            output_frequency_hz,
            self.TRAJECTORY_HORIZON_S,
            self._num_waypoints,
        )

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz

    def _encode_command(self, command: DriveCommand) -> int:
        """StaticModel does not steer, so the command is passed through."""
        return int(command)

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self._validate_cameras(prediction_input.camera_images)

        dt = 1.0 / self._output_frequency_hz
        times = np.arange(1, self._num_waypoints + 1, dtype=np.float64) * dt
        trajectory_xy = np.stack(
            [prediction_input.speed * times, np.zeros_like(times)], axis=-1
        )
        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=np.zeros(self._num_waypoints, dtype=np.float64),
        )
