# SPDX-License-Identifier: Apache-2.0

"""AutoVLA inference, trimmed to what the Alpasim driver needs.

Mirrors ``models.autovla.AutoVLA`` from ucla-mobility/AutoVLA (commit ba34eed)
with two deliberate differences:

* no ``pytorch_lightning`` / ``nuplan`` / ``navsim`` imports, since the Alpasim
  driver environment has none of them;
* camera frames arrive as in-memory ``PIL.Image`` objects rather than
  ``file://`` paths, so closed-loop stepping never touches the disk.

The prompt text is copied verbatim from ``AutoVLA.get_prompt`` — changing it
would take the model off the distribution it was fine-tuned on.
"""

from __future__ import annotations

import re
from importlib import resources

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .action_tokenizer import ActionTokenizer

# AutoVLA roles, in the order the prompt presents them.
CAMERA_ROLES = ("front_camera", "front_left_camera", "front_right_camera")

_VIEW_BLURB = {
    "front_camera": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz.",
    "front_left_camera": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz.",
    "front_right_camera": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz.",
}

SYSTEM_COT = (
    "You are an Advanced Driver Assistance and Full Self-Driving System. "
    "You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. "
    "Your task is to predict the optimal driving action for the next five seconds.\n\n"
    "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
    "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
    "Structure your reasoning as follows:\n"
    "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
    "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
    "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
    "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
    "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
    "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
    "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
    "Present the final action clearly after your reasoning steps."
)

SYSTEM_NOCOT = (
    "You are an Advanced Driver Assistance and Full Self-Driving System. "
    "You will be provided with video observations from the ego vehicle’s surrounding cameras, along with the vehicle’s current dynamic states. "
    "Your task is to predict the most appropriate driving action for the next five seconds."
)


def _apply_conversion_mapping(state_dict: dict, model_cls) -> dict:
    """Rename legacy Qwen2.5-VL keys to the layout the installed transformers wants.

    AutoVLA's checkpoint was written under transformers 4.49, which laid the
    weights out as ``visual.*`` / ``model.*`` / ``lm_head.*``. transformers 4.52
    moved them under ``model.visual.*`` and ``model.language_model.*``.
    ``from_pretrained`` applies ``_checkpoint_conversion_mapping`` for exactly
    this; ``load_state_dict`` does not, so apply it by hand. Without this every
    key silently misses and the policy drives on the base Qwen weights.

    A no-op on transformers versions that predate the mapping.
    """
    mapping = getattr(model_cls, "_checkpoint_conversion_mapping", None)
    if not mapping:
        return state_dict
    converted = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in mapping.items():
            new_key, n_subs = re.subn(pattern, replacement, new_key)
            if n_subs:
                break
        converted[new_key] = value
    return converted


def default_codebook_path() -> str:
    """Path to the action codebook shipped inside this package."""
    return str(resources.files(__package__) / "data" / "agent_vocab.pkl")


class AutoVLAInference:
    """Loads AutoVLA once and answers one driving query at a time."""

    # Fixed by the released checkpoint; not worth exposing as config.
    ACTION_START_ID = 151665
    NUM_POSES = 10
    INTERVAL_LENGTH_S = 0.5
    CONTEXT_LENGTH = 4
    MIN_PIXELS = 109760
    MAX_PIXELS = 109760

    def __init__(
        self,
        pretrained_model_path: str,
        checkpoint_path: str,
        device: torch.device | str = "cuda",
        use_cot: bool = True,
        codebook_path: str | None = None,
        max_length: int = 2048,
        temperature: float = 0.01,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> None:
        self._device = torch.device(device) if isinstance(device, str) else device
        self.use_cot = use_cot
        self._gen = {
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_path,
            dtype=torch.bfloat16,
            device_map=str(self._device),
        )
        self.processor = AutoProcessor.from_pretrained(pretrained_model_path)
        self.action_tokenizer = ActionTokenizer(
            self.processor.tokenizer,
            codebook_path=codebook_path or default_codebook_path(),
            action_start_id=self.ACTION_START_ID,
        )
        # AutoVLA appends one token per codebook entry before loading weights.
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))

        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = {
            key[len("autovla.vlm.") :]: value
            for key, value in state_dict.items()
            if key.startswith("autovla.vlm.")
        }
        if not state_dict:
            raise ValueError(
                f"No 'autovla.vlm.*' weights in {checkpoint_path}; "
                "is this an AutoVLA Lightning checkpoint?"
            )
        state_dict = _apply_conversion_mapping(
            state_dict, Qwen2_5_VLForConditionalGeneration
        )
        missing, unexpected = self.vlm.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"AutoVLA checkpoint did not match the model: {len(missing)} missing "
                f"and {len(unexpected)} unexpected keys (first missing: {missing[:3]}, "
                f"first unexpected: {unexpected[:3]})."
            )
        self.vlm.eval()

    def _build_messages(
        self,
        frames: dict[str, list[Image.Image]],
        velocity: float,
        acceleration: float,
        instruction: str,
    ) -> list[dict]:
        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, "
                    "left, and right, enabling a comprehensive perception of the surrounding environment."
                ),
            }
        ]
        for role in CAMERA_ROLES:
            images = frames[role]
            if len(images) != self.CONTEXT_LENGTH:
                raise ValueError(
                    f"{role}: AutoVLA expects {self.CONTEXT_LENGTH} frames, got {len(images)}"
                )
            user_content.append({"type": "text", "text": _VIEW_BLURB[role]})
            user_content.append(
                {
                    "type": "video",
                    "min_pixels": self.MIN_PIXELS,
                    "max_pixels": self.MAX_PIXELS,
                    "video": list(images),
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current "
                    f"acceleration is {acceleration:.3f} m/s². The driving instruction is: "
                    f"{instruction}. Based on this information, plan the action trajectory for the "
                    "autonomous vehicle over the next five seconds."
                ),
            }
        )
        system_text = SYSTEM_COT if self.use_cot else SYSTEM_NOCOT
        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    @torch.no_grad()
    def predict(
        self,
        frames: dict[str, list[Image.Image]],
        velocity: float,
        acceleration: float,
        instruction: str,
    ) -> tuple[np.ndarray, str]:
        """Plan 5 s ahead.

        Returns:
            ``(poses, text)`` where ``poses`` is ``(10, 3)`` of ``(x, y, heading)``
            in the rig frame at 0.5 s spacing, and ``text`` is the model's
            ``<think>…</think><answer>…</answer>`` output.
        """
        messages = self._build_messages(frames, velocity, acceleration, instruction)
        # `process_vision_info` accepts PIL images inside a "video" list, so the
        # frames go straight from the renderer to the processor.
        _, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        inputs = self.processor(
            text=[text],
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

        outputs = self.vlm.generate(
            **model_inputs,
            max_length=self._gen["max_length"],
            do_sample=True,
            temperature=self._gen["temperature"],
            top_k=self._gen["top_k"],
            top_p=self._gen["top_p"],
        )
        # Drop the prompt, then the trailing end-of-turn token.
        generated = outputs[0][len(inputs.input_ids[0]) :][:-1].cpu()
        reasoning_text = self.processor.decode(generated)

        action_tokens = generated[generated >= self.ACTION_START_ID]
        poses = self.action_tokenizer.decode_token_ids_to_trajectory(action_tokens)
        if len(poses) == 0:
            raise ValueError(
                "AutoVLA emitted no action tokens; decoded text: "
                f"{reasoning_text[:200]!r}"
            )
        return np.asarray(poses[0, 1:], dtype=np.float32), reasoning_text
