# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging
import os
import subprocess
import tempfile

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from eval.aggregation.processing import ProcessedMetricDFs
from eval.data import SimulationResult
from eval.schema import EvalConfig

logger = logging.getLogger("alpasim.eval.video_reasoning_overlay_utils")

mpl.use("Agg")
mplstyle.use("fast")


def _wrap_text(text: str, max_chars_per_line: int = 50) -> str:
    """Wrap text to fit within specified character width."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_chars_per_line:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def _overlay_reasoning_on_frame(frame, reasoning_text, time_s):
    """Overlay reasoning text on a video frame.

    Args:
        frame: numpy array (H, W, 3) uint8
        reasoning_text: The chain-of-thought reasoning text
        time_s: Current time in seconds

    Returns:
        numpy array with text overlay (H, W, 3)
    """
    # Convert to PIL Image
    img = Image.fromarray(frame)

    # Try to use a nicer font, fall back to default
    try:
        time_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32
        )
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32
        )
        reasoning_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30
        )
    except Exception:
        time_font = ImageFont.load_default()
        title_font = time_font
        reasoning_font = time_font

    h, w = frame.shape[:2]

    # Create RGBA overlay for semi-transparent backgrounds
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Time indicator at top left
    time_text = f"t = {time_s:.1f}s"
    overlay_draw.rectangle([(10, 10), (190, 58)], fill=(0, 0, 0, 200))
    overlay_draw.text((20, 12), time_text, fill=(255, 255, 255), font=time_font)

    # Reasoning text - prominent display at TOP of frame (below time indicator)
    if reasoning_text:
        # Calculate max chars that fit based on frame width
        max_chars = (w - 80) // 18  # Leave 40px padding on each side
        wrapped_text = _wrap_text(reasoning_text, max_chars_per_line=max_chars)
        lines = wrapped_text.split("\n")
        line_height = 40
        title_height = 45
        padding = 20
        box_height = title_height + len(lines) * line_height + padding * 2
        box_top = 65  # Position below time indicator

        # Draw background box with accent border
        overlay_draw.rectangle(
            [(10, box_top), (w - 10, box_top + box_height)], fill=(20, 20, 40, 220)
        )
        # Top accent line (cyan/teal to match trajectory prediction color)
        overlay_draw.rectangle(
            [(10, box_top), (w - 10, box_top + 4)], fill=(78, 205, 196, 255)
        )

        # Composite overlay
        img = Image.alpha_composite(img.convert("RGBA"), overlay)
        draw = ImageDraw.Draw(img)

        # Draw title
        draw.text(
            (20, box_top + 10), "Reasoning:", fill=(78, 205, 196), font=title_font
        )

        # Draw reasoning text lines
        y = box_top + 10 + title_height
        for line in lines:
            draw.text((20, y), line, fill=(255, 255, 255), font=reasoning_font)
            y += line_height

        img = img.convert("RGB")
    else:
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    return np.array(img)


def _create_trajectory_chart_reasoning_overlay_style(
    ego_history_xyz, pred_xyz, figsize=(6, 6), dpi=100
):
    """Create an ego-centric trajectory chart showing history and prediction (reasoning overlay style).

    Args:
        ego_history_xyz: History trajectory in ego frame, numpy array (T_hist, 3) or tensor
        pred_xyz: Predicted trajectory in ego frame, numpy array (T_future, 3) or tensor
        figsize: Figure size
        dpi: DPI for the figure

    Returns:
        numpy array of the chart image (H, W, 3)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    def prepare_ego_frame_coords_for_plot(xyz_array):
        """Convert ego-centric coordinates to plot coordinates."""
        x_ego = xyz_array[:, 0]  # longitudinal
        y_ego = xyz_array[:, 1]  # lateral
        plot_x = -y_ego  # lateral on x-axis
        plot_y = x_ego  # longitudinal on y-axis
        return plot_x, plot_y

    # Convert torch tensors to numpy if needed
    if hasattr(ego_history_xyz, "cpu"):
        hist_xyz = ego_history_xyz.cpu().numpy()
    else:
        hist_xyz = np.array(ego_history_xyz)

    # Handle shape: expect (1, 1, T, 3) or (T, 3)
    if hist_xyz.ndim == 4:
        hist_xyz = hist_xyz[0, 0, :, :]
    elif hist_xyz.ndim == 2:
        hist_xyz = hist_xyz
    else:
        raise ValueError(f"Unexpected history shape: {hist_xyz.shape}")

    # Plot history trajectory if not empty
    if len(hist_xyz) > 0:
        plot_x, plot_y = prepare_ego_frame_coords_for_plot(hist_xyz)
        ax.plot(
            plot_x,
            plot_y,
            "o-",
            color="#888888",
            linewidth=2,
            markersize=4,
            label="History",
            alpha=0.7,
        )

    # Plot predicted trajectory
    if pred_xyz is not None:
        if hasattr(pred_xyz, "cpu"):
            pred_xyz_np = pred_xyz.cpu().numpy()
        else:
            pred_xyz_np = np.array(pred_xyz)

        # Handle various shapes
        if pred_xyz_np.ndim == 5:
            num_samples = pred_xyz_np.shape[2]
            for i in range(num_samples):
                pred_xyz_sample = pred_xyz_np[0, 0, i, :, :]
                if len(pred_xyz_sample) > 0:
                    plot_x, plot_y = prepare_ego_frame_coords_for_plot(pred_xyz_sample)
                    ax.plot(
                        plot_x,
                        plot_y,
                        "o-",
                        color="#4ecdc4",
                        linewidth=2.5,
                        markersize=5,
                        label="Prediction" if i == 0 else None,
                        alpha=0.9,
                    )
        elif pred_xyz_np.ndim == 3:
            num_samples = pred_xyz_np.shape[0]
            for i in range(num_samples):
                pred_xyz_sample = pred_xyz_np[i, :, :]
                if len(pred_xyz_sample) > 0:
                    plot_x, plot_y = prepare_ego_frame_coords_for_plot(pred_xyz_sample)
                    ax.plot(
                        plot_x,
                        plot_y,
                        "o-",
                        color="#4ecdc4",
                        linewidth=2.5,
                        markersize=5,
                        label="Prediction" if i == 0 else None,
                        alpha=0.9,
                    )
        elif pred_xyz_np.ndim == 2 and len(pred_xyz_np) > 0:
            plot_x, plot_y = prepare_ego_frame_coords_for_plot(pred_xyz_np)
            ax.plot(
                plot_x,
                plot_y,
                "o-",
                color="#4ecdc4",
                linewidth=2.5,
                markersize=5,
                label="Prediction",
                alpha=0.9,
            )

    # Mark current position (ego at t0) - rectangle to represent car shape
    car_length = 4.5
    car_width = 1.8
    from matplotlib.patches import Rectangle

    ego_rect = Rectangle(
        (-car_width / 2, -car_length / 2),
        car_width,
        car_length,
        facecolor="#ffd93d",
        edgecolor="#ffffff",
        linewidth=1.5,
        zorder=5,
        label="Ego Vehicle",
    )
    ax.add_patch(ego_rect)

    # Set FIXED axis limits and ticks
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 80)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_aspect("equal", adjustable="box")

    # Styling
    ax.set_xlabel("Lateral (m)", fontsize=12, color="white")
    ax.set_ylabel("Longitudinal (m)", fontsize=12, color="white")
    ax.set_title(
        "Trajectory Prediction", fontsize=14, fontweight="bold", color="white", pad=10
    )
    ax.legend(
        loc="upper right",
        fontsize=10,
        facecolor="#2d2d44",
        edgecolor="white",
        labelcolor="white",
    )
    ax.grid(True, alpha=0.3, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()

    # Convert figure to numpy array
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB

    plt.close(fig)
    return img


def _combine_panes(left_frame, right_frame):
    """Combine left and right panes into a single frame.

    Args:
        left_frame: Left pane image (H, W, 3)
        right_frame: Right pane image (H, W, 3)

    Returns:
        Combined frame (H, W*2, 3)
    """
    # Resize both to same height
    target_height = max(left_frame.shape[0], right_frame.shape[0])

    # Resize left frame
    if left_frame.shape[0] != target_height:
        scale = target_height / left_frame.shape[0]
        new_width = int(left_frame.shape[1] * scale)
        left_frame = cv2.resize(left_frame, (new_width, target_height))

    # Resize right frame to match height
    if right_frame.shape[0] != target_height:
        scale = target_height / right_frame.shape[0]
        new_width = int(right_frame.shape[1] * scale)
        right_frame = cv2.resize(right_frame, (new_width, target_height))

    # Combine horizontally
    combined = np.concatenate([left_frame, right_frame], axis=1)
    return combined


def _save_video_with_ffmpeg(
    frames: list[np.ndarray], output_path: str, fps: int = 10
) -> None:
    """Save video using FFmpeg subprocess as fallback when OpenCV fails.

    Args:
        frames: List of RGB frames (numpy arrays)
        output_path: Path to save the video
        fps: Frames per second
    """
    logger.info(f"Using FFmpeg to write {len(frames)} frames to {output_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("Saving frames as PNG files...")
        for idx, frame in enumerate(tqdm(frames, desc="Saving frames")):
            frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)

        logger.info("Creating video with FFmpeg...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(temp_dir, "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "medium",
            output_path,
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            logger.info(f"FFmpeg video saved successfully to: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with error: {e.stderr}")
            raise RuntimeError(f"Failed to create video with FFmpeg: {e.stderr}")
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg.")
            raise RuntimeError("FFmpeg is not installed or not in PATH")


def _write_video(frames: list[np.ndarray], output_path: str, fps: int = 10) -> None:
    """Write RGB frames to an mp4, preferring OpenCV and falling back to FFmpeg.

    OpenCV ships its own encoder, so this works on hosts without a system FFmpeg.
    """
    if not frames:
        logger.warning("No frames to render")
        return

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        logger.warning("OpenCV MPEG-4 writer unavailable, falling back to FFmpeg")
        _save_video_with_ffmpeg(frames, output_path, fps)
        return

    logger.info(
        "Writing %d frames at %dx%d to %s", len(frames), width, height, output_path
    )
    for idx, frame in enumerate(tqdm(frames, desc="Writing video")):
        assert frame.shape[:2] == (height, width), (
            f"Frame {idx} has shape {frame.shape[:2]}, expected ({height}, {width})"
        )
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    logger.info("Video saved to: %s", output_path)


def _get_camera_frame_numpy(camera, time_us) -> np.ndarray:
    """Extract camera frame as numpy array.

    Args:
        camera: Camera object
        time_us: Timestamp in microseconds

    Returns:
        Camera frame as numpy array (H, W, 3) uint8
    """
    pil_image = camera.image_at_time(time_us)

    if pil_image is not None:
        img_array = np.array(pil_image)

        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)

        return img_array
    else:
        return np.zeros((480, 640, 3), dtype=np.uint8)


def _render_single_reasoning_overlay_frame(
    sim_result: SimulationResult,
    time_us: np.uint64,
    cached_reasoning: str,
    display_time_s: float,
    cfg: EvalConfig,
) -> np.ndarray:
    """Render a single reasoning overlay style frame.

    Args:
        sim_result: The simulation result to render.
        time_us: Timestamp in microseconds to render
        cached_reasoning: Cached CoT reasoning text to display
        display_time_s: Display time in seconds (for overlay)
        cfg: Evaluation configuration

    Returns:
        Combined frame as numpy array (H, W, 3)
    """
    time_us_int = int(time_us)
    camera = sim_result.cameras.camera_by_logical_id[cfg.video.camera_id_to_render]
    driver_responses = sim_result.driver_responses

    # Get camera frame
    camera_frame = _get_camera_frame_numpy(camera, time_us_int)

    # Create left pane: camera with reasoning overlay
    left_frame = _overlay_reasoning_on_frame(
        camera_frame, cached_reasoning, display_time_s
    )

    # Get trajectory data at this timestamp
    driver_response_at_time = driver_responses.get_driver_response_for_time(
        time_us_int, which_time="now"
    )

    # Create right pane: trajectory chart
    if driver_response_at_time is None:
        logger.debug(f"No driver response at time {time_us}us, using dummy trajectory")
        ego_history_xyz = np.zeros((0, 3))
        pred_xyz_np = np.zeros((0, 3))
    else:
        ego_traj = sim_result.driver_estimated_trajectory
        gt_timestamps = ego_traj.timestamps_us
        gt_start_us = gt_timestamps[0] if len(gt_timestamps) > 0 else 0

        ego_pose_at_time = ego_traj.interpolate_to_timestamps(
            np.array([time_us_int], dtype=np.uint64)
        )
        ego_qvec = ego_pose_at_time.get_pose(0)
        ego_xyz = ego_qvec.vec3

        pred_xyz_np = driver_response_at_time.selected_trajectory.positions
        if np.any(np.abs(pred_xyz_np[0] - ego_xyz) > np.array([0.01, 0.01, 0.05])):
            raise ValueError(
                "First predicted point is not close to current ego position: "
                f"{pred_xyz_np[0]} - {ego_xyz} = {np.abs(pred_xyz_np[0] - ego_xyz)}"
            )

        # Get history
        history_duration_us = 1_600_000  # time window for visualization
        history_start_us = max(time_us_int - history_duration_us, int(gt_start_us))
        history_timestamps = np.arange(
            history_start_us, time_us_int, 100_000, dtype=np.uint64
        )
        ego_history = ego_traj.interpolate_to_timestamps(history_timestamps)
        ego_history_xyz = ego_history.positions

        # Transform to ego frame
        world_to_ego_transform = ego_qvec.inverse().as_se3()
        pred_homogeneous = np.hstack([pred_xyz_np, np.ones((pred_xyz_np.shape[0], 1))])
        pred_xyz_np = (world_to_ego_transform @ pred_homogeneous.T).T[:, :3]
        ego_history_homogeneous = np.hstack(
            [ego_history_xyz, np.ones((ego_history_xyz.shape[0], 1))]
        )
        ego_history_xyz = (world_to_ego_transform @ ego_history_homogeneous.T).T[:, :3]

    right_frame = _create_trajectory_chart_reasoning_overlay_style(
        ego_history_xyz, pred_xyz_np, figsize=(6, 6), dpi=100
    )
    combined_frame = _combine_panes(left_frame, right_frame)

    return combined_frame


def render_reasoning_overlay_style_video(
    sim_result: SimulationResult,
    processed_metric_dfs: ProcessedMetricDFs,
    output_path: str,
    cfg: EvalConfig,
) -> None:
    """Render video in reasoning overlay style:
        first-person camera with reasoning text overlay + trajectory chart on the right.

    Args:
        sim_result: The simulation result to render.
        processed_metric_dfs: Processed metrics dataframes
        output_path: Path to save the video
        cfg: Evaluation configuration
    """
    del processed_metric_dfs
    logger.info("Rendering reasoning overlay style video")

    timestamps_us = sim_result.timestamps_us
    driver_responses = sim_result.driver_responses

    # Render every nth frame as configured
    timestamps_to_render = timestamps_us[:: cfg.video.render_every_nth_frame]

    # CoT refresh interval
    reasoning_text_refresh_interval_s = cfg.video.reasoning_text_refresh_interval_s
    if reasoning_text_refresh_interval_s is None:
        reasoning_text_refresh_interval_s = 0.0  # Refresh every frame

    output_frames = []
    cached_reasoning = ""
    last_reasoning_text_refresh_time_us = (
        timestamps_to_render[0] if len(timestamps_to_render) > 0 else 0
    )

    expected_frame_shape = None

    # Process frames
    for frame_idx, time_us in enumerate(
        tqdm(timestamps_to_render, desc="Rendering frames")
    ):
        display_time_s = (time_us - timestamps_to_render[0]) / 1_000_000.0

        # Check if we should update CoT
        should_update_reasoning_text = (
            time_us - last_reasoning_text_refresh_time_us
        ) >= reasoning_text_refresh_interval_s * 1_000_000

        if should_update_reasoning_text and time_us in driver_responses.timestamps_us:
            idx = driver_responses.timestamps_us.index(time_us)
            driver_response_at_time = driver_responses.per_timestep_driver_responses[
                idx
            ]
            if driver_response_at_time.reasoning_text is not None:
                cached_reasoning = driver_response_at_time.reasoning_text
                last_reasoning_text_refresh_time_us = time_us
                logger.debug(
                    f"Updated reasoning text at time {time_us}us: {len(cached_reasoning)} chars"
                )

        # Render the frame
        combined_frame = _render_single_reasoning_overlay_frame(
            sim_result,
            time_us,
            cached_reasoning,
            display_time_s,
            cfg,
        )

        # Assert frame shape consistency
        if expected_frame_shape is None:
            expected_frame_shape = combined_frame.shape
            logger.info(f"Frame shape established: {expected_frame_shape}")
        else:
            assert combined_frame.shape == expected_frame_shape, (
                f"Frame {frame_idx} has shape {combined_frame.shape}, "
                f"expected {expected_frame_shape}"
            )

        output_frames.append(combined_frame)

    _write_video(output_frames, output_path, fps=10)


# Tile order for the 2x2 quad-cam view: forward-facing on top, cross views below.
QUAD_CAM_ORDER = (
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
)


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    """Draw a camera name in the bottom-left corner of a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()

    h = frame.shape[0]
    draw.rectangle([(0, h - 30), (len(label) * 12 + 16, h)], fill=(0, 0, 0))
    draw.text((8, h - 27), label, fill=(255, 255, 255), font=font)
    return np.array(img)


def _tile_quad(
    sim_result: SimulationResult, time_us: int, tile_size: tuple[int, int]
) -> np.ndarray:
    """Tile the four driver cameras into a single 2x2 frame.

    Cameras absent from the rollout are rendered as black tiles so the output
    frame size stays constant, which the video writer requires.
    """
    tile_w, tile_h = tile_size
    tiles = []
    for camera_id in QUAD_CAM_ORDER:
        camera = sim_result.cameras.camera_by_logical_id.get(camera_id)
        if camera is None:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            continue
        frame = _get_camera_frame_numpy(camera, time_us)
        frame = cv2.resize(frame, (tile_w, tile_h))
        tiles.append(_label_frame(frame, camera_id))

    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:])
    return np.vstack([top, bottom])


def _overlay_nav_text(frame: np.ndarray, nav_text: str | None) -> np.ndarray:
    """Draw the navigation instruction the model was given, in the top-right.

    Renders "NAV: none" when the driver was not navigation-conditioned, so the
    video never implies the model received an instruction it did not.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30
        )
    except Exception:
        font = ImageFont.load_default()

    text = f"NAV: {nav_text}" if nav_text else "NAV: none"
    w = frame.shape[1]
    text_w = draw.textlength(text, font=font)
    padding = 12
    box_left = w - 10 - text_w - 2 * padding
    draw.rectangle([(box_left, 10), (w - 10, 58)], fill=(0, 0, 0))
    draw.text((box_left + padding, 14), text, fill=(120, 220, 255), font=font)
    return np.array(img)


def render_quad_cam_video(
    sim_result: SimulationResult,
    output_path: str,
    cfg: EvalConfig,
) -> None:
    """Render all four driver cameras tiled 2x2, with reasoning and nav text.

    Each tile is rendered at half the reference camera's resolution, so the
    output video is the same size as the single-camera layouts.

    Args:
        sim_result: The simulation result to render.
        output_path: Path to save the video.
        cfg: Evaluation configuration.
    """
    logger.info("Rendering quad-cam video")

    timestamps_us = sim_result.timestamps_us
    driver_responses = sim_result.driver_responses
    timestamps_to_render = timestamps_us[:: cfg.video.render_every_nth_frame]
    if len(timestamps_to_render) == 0:
        logger.warning("No frames to render")
        return

    reference = sim_result.cameras.camera_by_logical_id[cfg.video.camera_id_to_render]
    reference_frame = _get_camera_frame_numpy(reference, timestamps_to_render[0])
    tile_size = (reference_frame.shape[1] // 2, reference_frame.shape[0] // 2)

    reasoning_refresh_interval_s = cfg.video.reasoning_text_refresh_interval_s or 0.0

    output_frames = []
    cached_reasoning = ""
    cached_nav_text = None
    last_reasoning_refresh_us = timestamps_to_render[0]

    for time_us in tqdm(timestamps_to_render, desc="Rendering quad-cam frames"):
        display_time_s = (time_us - timestamps_to_render[0]) / 1_000_000.0

        if time_us in driver_responses.timestamps_us:
            idx = driver_responses.timestamps_us.index(time_us)
            response = driver_responses.per_timestep_driver_responses[idx]
            cached_nav_text = response.nav_text
            should_refresh = (
                time_us - last_reasoning_refresh_us
            ) >= reasoning_refresh_interval_s * 1_000_000
            if should_refresh and response.reasoning_text is not None:
                cached_reasoning = response.reasoning_text
                last_reasoning_refresh_us = time_us

        frame = _tile_quad(sim_result, time_us, tile_size)
        frame = _overlay_reasoning_on_frame(frame, cached_reasoning, display_time_s)
        frame = _overlay_nav_text(frame, cached_nav_text)
        output_frames.append(frame)

    _write_video(output_frames, output_path, fps=10)
