# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Navigation utilities for determining driving commands from route geometry."""

from __future__ import annotations

import logging

import numpy as np
from alpasim_grpc.v0.common_pb2 import Vec3
from alpasim_grpc.v0.egodriver_pb2 import Route

from .models.base import DriveCommand

logger = logging.getLogger(__name__)

# Shorter than this, the first route segment gives no reliable path direction.
_MIN_TANGENT_LENGTH_M = 1e-3


def _find_target_waypoint(route: Route, min_lookahead_distance: float) -> "Vec3 | None":
    """Return the first waypoint at least ``min_lookahead_distance`` ahead."""
    for wp in route.waypoints:
        if np.hypot(wp.x, wp.y) >= min_lookahead_distance:
            return wp
    return None


def determine_command_from_route(
    route: Route,
    command_distance_threshold: float = 2.0,
    min_lookahead_distance: float = 5.0,
) -> DriveCommand:
    """Determine semantic driving command from route geometry.

    Analyzes route waypoints (in rig frame) to determine whether the
    vehicle should turn left, go straight, or turn right.

    Args:
        route: Route containing waypoints in the rig frame.
        command_distance_threshold: Lateral distance threshold (meters) for
            determining turn commands. Waypoints beyond this threshold
            trigger LEFT/RIGHT commands.
        min_lookahead_distance: Minimum forward distance (meters) to consider
            a waypoint as the target for command derivation.

    Returns:
        Semantic DriveCommand (LEFT, STRAIGHT, RIGHT, or UNKNOWN).
    """
    if len(route.waypoints) < 1:
        return DriveCommand.UNKNOWN

    target_waypoint = _find_target_waypoint(route, min_lookahead_distance)

    if target_waypoint is None:
        return DriveCommand.STRAIGHT

    # In rig frame, positive Y is left
    dy_rig = target_waypoint.y

    if dy_rig > command_distance_threshold:
        command = DriveCommand.LEFT
    elif dy_rig < -command_distance_threshold:
        command = DriveCommand.RIGHT
    else:
        command = DriveCommand.STRAIGHT

    logger.debug(
        "Command: %s (lateral displacement: %.2fm at distance: %.2fm)",
        command.name,
        dy_rig,
        np.hypot(target_waypoint.x, target_waypoint.y),
    )

    return command


def _waypoints_in_path_frame(route: Route) -> np.ndarray | None:
    """Re-express the route waypoints in the frame of the path itself.

    The waypoints arrive in the rig frame of the *simulated* ego, so a waypoint's
    lateral offset mixes two things: how the route bends, and how far the ego has
    drifted off it. Once the ego strays left of a straight route, that route sits
    to its right and offset-based logic calls for a right turn -- telling the
    model to compound the error it just made.

    The route generator already resamples the recorded path from the ego's
    projection onto it, so waypoints[0] is that projection and the first segment
    is the path's tangent there. Rotating about it puts x along the path, leaving
    y as offset *from the path*, which no longer depends on where the ego is.

    Returns None when the route is too short or degenerate to define a tangent.
    """
    waypoints = np.array([[wp.x, wp.y] for wp in route.waypoints])
    # Routes can carry non-finite waypoints; they would poison the tangent.
    waypoints = waypoints[np.isfinite(waypoints).all(axis=1)]
    if len(waypoints) < 2:
        return None

    tangent = waypoints[1] - waypoints[0]
    if np.hypot(*tangent) < _MIN_TANGENT_LENGTH_M:
        return None

    angle = -np.arctan2(tangent[1], tangent[0])
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    return (waypoints - waypoints[0]) @ rotation.T


def nav_text_from_route(
    route: Route,
    command_distance_threshold: float = 2.0,
    min_lookahead_distance: float = 5.0,
) -> str | None:
    """Build a natural-language navigation instruction from route geometry.

    Phrasing matches what Alpamayo 1.5 was trained on: "Turn left in 30m" for
    turns, "Continue straight" otherwise. Announcing the straight case matters:
    without it the model has no signal to suppress a turn it invents on its own
    at an intersection.

    Applies the same rule as determine_command_from_route -- the first waypoint
    past the lookahead, offset further than the threshold -- but in the path frame
    rather than the rig frame, so the instruction describes the road ahead rather
    than the ego's drift off it. The distance is to where the route begins to
    depart, so it counts down as the ego closes on the turn.

    Returns None when the route is too short to have a shape, in which case the
    model is left unconditioned.

    Args:
        route: Route containing waypoints in the rig frame.
        command_distance_threshold: Lateral distance threshold (meters) beyond
            which the route counts as a turn.
        min_lookahead_distance: Minimum forward distance (meters) to consider a
            waypoint as the turn target.
    """
    waypoints = _waypoints_in_path_frame(route)
    if waypoints is None:
        return None

    distances = np.hypot(waypoints[:, 0], waypoints[:, 1])
    ahead = np.flatnonzero(distances >= min_lookahead_distance)
    if len(ahead) == 0:
        return "Continue straight"

    # Positive y is left of the path
    target_offset = waypoints[ahead[0], 1]
    if target_offset > command_distance_threshold:
        direction = "left"
    elif target_offset < -command_distance_threshold:
        direction = "right"
    else:
        return "Continue straight"

    departs = np.flatnonzero(
        np.abs(waypoints[:, 1]) > command_distance_threshold
    )
    onset = int(departs[0]) if len(departs) else int(ahead[0])
    return f"Turn {direction} in {round(float(distances[onset]))}m"
