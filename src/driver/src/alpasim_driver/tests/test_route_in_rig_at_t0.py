# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Route waypoints re-expressed in the rig frame at t0."""

from __future__ import annotations

import numpy as np
from alpasim_driver.models.alpamayo_base import route_in_rig_at_t0
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Vec3
from alpasim_grpc.v0.egodriver_pb2 import Route


def _pose(t_us: int, x: float, y: float, yaw: float) -> PoseAtTime:
    return PoseAtTime(
        timestamp_us=t_us,
        pose=Pose(
            vec=Vec3(x=x, y=y, z=0.0),
            quat=Quat(x=0.0, y=0.0, z=np.sin(yaw / 2), w=np.cos(yaw / 2)),
        ),
    )


def test_ego_motion_since_route_is_taken_out() -> None:
    # Ego drives 2 m forward along local +y (yaw 90 deg) between route and t0.
    poses = [_pose(0, 0.0, 0.0, np.pi / 2), _pose(1_000_000, 0.0, 2.0, np.pi / 2)]
    route = Route(
        timestamp_us=0,
        waypoints=[Vec3(x=10.0, y=0.0, z=0.0), Vec3(x=10.0, y=5.0, z=0.0)],
    )

    waypoints = route_in_rig_at_t0(route, poses, t0_us=1_000_000)

    np.testing.assert_allclose(waypoints[:, :2], [[8.0, 0.0], [8.0, 5.0]], atol=1e-4)


def test_route_at_t0_is_unchanged() -> None:
    poses = [_pose(0, 3.0, 1.0, 0.3), _pose(1_000_000, 5.0, 2.0, 0.4)]
    route = Route(timestamp_us=1_000_000, waypoints=[Vec3(x=7.0, y=-2.0, z=0.0)])

    waypoints = route_in_rig_at_t0(route, poses, t0_us=1_000_000)

    np.testing.assert_allclose(waypoints, [[7.0, -2.0, 0.0]], atol=1e-4)
