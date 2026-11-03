# SPDX-License-Identifier: Apache-2.0

"""AutoVLA wrapper implementing the Alpasim driver model interface."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.schema import ModelConfig
from PIL import Image

from .inference import CAMERA_ROLES, AutoVLAInference

logger = logging.getLogger(__name__)

# Alpasim logical camera id -> the role AutoVLA's prompt gives it.
#
# AutoVLA was trained on nuPlan CAM_F0 / CAM_L0 / CAM_R0. Alpasim's cross
# cameras are 120 deg where nuPlan's L0/R0 are ~70 deg, so the side views are
# wider than what the policy saw in training; see README.md.
DEFAULT_CAMERA_ROLES = {
    "camera_front_wide_120fov": "front_camera",
    "camera_cross_left_120fov": "front_left_camera",
    "camera_cross_right_120fov": "front_right_camera",
}

# AutoVLA's nuPlan instruction vocabulary, from navsim/navsim/agents/vla_agent.py.
# The prompt lowercases these, matching the preprocessing in
# dataset_utils/preprocessing/nuplan_dataset.py.
COMMAND_TEXT = {
    DriveCommand.LEFT: "turn left",
    DriveCommand.STRAIGHT: "keep forward",
    DriveCommand.RIGHT: "turn right",
    DriveCommand.UNKNOWN: "unknown",
}


def _route_lateral_profile(route: object | None) -> str:
    """Lateral offset of the route at a few lookaheads, for calibrating commands.

    alpasim derives LEFT/RIGHT from the first waypoint past
    ``min_lookahead_distance`` exceeding ``command_distance_threshold``
    laterally. Seeing the whole profile tells us whether a given threshold is
    reacting to an actual turn or merely to road curvature.
    """
    waypoints = getattr(route, "waypoints", None)
    if not waypoints:
        return "none"
    parts = []
    for lookahead in (10.0, 20.0, 40.0, 60.0):
        for waypoint in waypoints:
            if np.hypot(waypoint.x, waypoint.y) >= lookahead:
                parts.append(f"{lookahead:.0f}m:{waypoint.y:+.1f}")
                break
        else:
            parts.append(f"{lookahead:.0f}m:-")
    return ",".join(parts)


class AutoVLAModel(BaseTrajectoryModel):
    """AutoVLA (Qwen2.5-VL-3B + action tokens) as an Alpasim driver."""

    # AutoVLA consumes 4 frames per camera at 2 Hz and plans 10 poses at 0.5 s.
    CONTEXT_LENGTH = 4
    OUTPUT_FREQUENCY_HZ = 2

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "AutoVLAModel":
        """Create AutoVLAModel from driver configuration.

        ``tokenizer_path`` carries the Qwen2.5-VL-3B-Instruct directory that the
        checkpoint was fine-tuned from — the driver schema has no dedicated
        field for a base model, and this is the same slot VAM uses for its
        tokenizer.
        """
        if model_cfg.tokenizer_path is None:
            raise ValueError(
                "AutoVLA needs tokenizer_path set to the Qwen2.5-VL-3B-Instruct directory"
            )
        if context_length not in (None, cls.CONTEXT_LENGTH):
            raise ValueError(
                f"AutoVLA requires context_length={cls.CONTEXT_LENGTH}, got {context_length}"
            )
        if output_frequency_hz != cls.OUTPUT_FREQUENCY_HZ:
            raise ValueError(
                f"AutoVLA plans at {cls.OUTPUT_FREQUENCY_HZ} Hz, "
                f"got output_frequency_hz={output_frequency_hz}"
            )
        return cls(
            pretrained_model_path=model_cfg.tokenizer_path,
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
        )

    def __init__(
        self,
        pretrained_model_path: str,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        use_cot: bool = True,
    ) -> None:
        unknown = [c for c in camera_ids if c not in DEFAULT_CAMERA_ROLES]
        if unknown:
            raise ValueError(
                f"AutoVLA has no role for camera(s) {unknown}; "
                f"expected a subset of {sorted(DEFAULT_CAMERA_ROLES)}"
            )
        roles = {DEFAULT_CAMERA_ROLES[c] for c in camera_ids}
        if roles != set(CAMERA_ROLES):
            raise ValueError(
                f"AutoVLA needs all three of {CAMERA_ROLES}; "
                f"cameras {camera_ids} only cover {sorted(roles)}"
            )

        self._camera_ids = list(camera_ids)
        self._role_to_camera = {DEFAULT_CAMERA_ROLES[c]: c for c in camera_ids}
        self._debug_dir = os.environ.get("AUTOVLA_DEBUG_DIR")
        self._step = 0

        logger.info(
            "Loading AutoVLA: base=%s checkpoint=%s device=%s",
            pretrained_model_path,
            checkpoint_path,
            device,
        )
        self._model = AutoVLAInference(
            pretrained_model_path=pretrained_model_path,
            checkpoint_path=checkpoint_path,
            device=device,
            use_cot=use_cot,
        )
        logger.info("AutoVLA ready (cameras: %s)", self._camera_ids)

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self.CONTEXT_LENGTH

    @property
    def output_frequency_hz(self) -> int:
        return self.OUTPUT_FREQUENCY_HZ

    def _encode_command(self, command: DriveCommand) -> str:
        """Convert DriveCommand to the instruction string AutoVLA was trained on."""
        return COMMAND_TEXT[command]

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self._validate_cameras(prediction_input.camera_images)

        frames: dict[str, list[Image.Image]] = {}
        for role in CAMERA_ROLES:
            camera_frames = prediction_input.camera_images[self._role_to_camera[role]]
            if len(camera_frames) != self.CONTEXT_LENGTH:
                raise ValueError(
                    f"{role}: AutoVLA expects {self.CONTEXT_LENGTH} frames, "
                    f"got {len(camera_frames)}"
                )
            # Frames arrive oldest-first as plain (timestamp_us, HWC uint8 RGB)
            # tuples — the driver builds them positionally even though the type
            # is annotated CameraFrame, so unpack rather than use attributes.
            frames[role] = [Image.fromarray(image) for _timestamp_us, image in camera_frames]

        instruction = self._encode_command(prediction_input.command)
        speed = float(prediction_input.speed)
        acceleration = float(prediction_input.acceleration)

        poses, reasoning_text = self._model.predict(
            frames=frames,
            velocity=speed,
            acceleration=acceleration,
            instruction=instruction,
        )

        # The conditioning is the first thing to check when the ego wanders:
        # a flapping or stuck command steers the policy regardless of the view.
        logger.info(
            "step=%d command=%s speed=%.2f accel=%.2f endpoint=(%.1f, %.1f) cot=%s route=%s",
            self._step,
            instruction,
            speed,
            acceleration,
            poses[-1, 0],
            poses[-1, 1],
            "yes" if "<think>" in reasoning_text and len(reasoning_text) > 300 else "no",
            _route_lateral_profile(prediction_input.route),
        )

        if self._debug_dir:
            self._dump_debug(
                frames,
                poses,
                reasoning_text,
                instruction,
                speed,
                acceleration,
                prediction_input.route,
            )
        self._step += 1

        trajectory_xy = np.ascontiguousarray(poses[:, :2], dtype=np.float64)
        headings = np.ascontiguousarray(poses[:, 2], dtype=np.float64)
        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning_text,
        )

    def _dump_debug(
        self,
        frames: dict[str, list[Image.Image]],
        poses: np.ndarray,
        reasoning_text: str,
        instruction: str,
        speed: float,
        acceleration: float,
        route: object | None = None,
    ) -> None:
        """Dump everything one inference saw and produced.

        Each step gets the newest frame per camera plus a ``step.json`` holding
        the conditioning, the decoded plan and the reasoning text, which is what
        ``tools/visualize_autovla_steps.py`` renders.
        """
        out_dir = os.path.join(self._debug_dir, f"step_{self._step:04d}")
        os.makedirs(out_dir, exist_ok=True)
        for role, images in frames.items():
            images[-1].save(os.path.join(out_dir, f"{role}.jpg"), quality=85)

        waypoints = getattr(route, "waypoints", None) or []
        record = {
            "step": self._step,
            "instruction": instruction,
            "speed_mps": speed,
            "acceleration_mps2": acceleration,
            "reasoning_text": reasoning_text,
            # (x, y, heading) in the rig frame, 0.5 s apart, 10 poses over 5 s
            "poses": [[float(v) for v in row] for row in poses],
            # route waypoints in the rig frame at the route's own timestamp
            "route_xy": [[float(w.x), float(w.y)] for w in waypoints],
        }
        with open(os.path.join(out_dir, "step.json"), "w") as handle:
            json.dump(record, handle, indent=1)
