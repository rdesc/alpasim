# SPDX-License-Identifier: Apache-2.0
"""The teacher satnav copy (nav_satnav.instruction) and its wording in Alpamayo's phrasing."""
from __future__ import annotations

import numpy as np
import pytest

from alpasim_driver.models import nav_satnav


def _path(turn_at_m: float, turn_deg: float, length_m: float = 80.0) -> np.ndarray:
    """Straight for turn_at_m, then an arc to turn_deg (+ left) over 20 m, then straight."""
    pts, x, y, h = [], 0.0, 0.0, 0.0
    for s in np.arange(0.0, length_m, 1.0):
        if turn_at_m <= s < turn_at_m + 20.0:
            h += np.radians(turn_deg) / 20.0
        x += np.cos(h)
        y += np.sin(h)
        pts.append((x, y))
    return np.array(pts)


@pytest.mark.parametrize(
    "turn_deg, maneuver, text",
    [
        (0.0, "straight", "Continue straight"),
        (90.0, "turn left", "Turn left in 20m"),
        (-90.0, "turn right", "Turn right in 20m"),
        # a gentle bend passes the 8 deg start ~29 m in, so it is reported at 30 m
        (-18.0, "bear right", "Turn right in 30m"),
    ],
)
def test_turns_are_worded_in_alpamayo_phrasing(turn_deg, maneuver, text):
    instr = nav_satnav.instruction(_path(20.0, turn_deg))
    assert instr["maneuver"] == maneuver
    assert nav_satnav.alpamayo_text(instr) == text


def test_bear_can_be_worded_as_straight():
    instr = nav_satnav.instruction(_path(20.0, 18.0))
    assert instr["maneuver"] == "bear left"
    assert nav_satnav.alpamayo_text(instr, bear_as="straight") == "Continue straight"


def test_a_turn_under_way_is_reported_a_few_metres_out():
    instr = nav_satnav.instruction(_path(1.0, -90.0))
    assert instr["distance_m"] == 0
    assert nav_satnav.alpamayo_text(instr) == "Turn right in 5m"


def test_nan_padding_is_dropped_and_no_route_gives_no_text():
    padded = np.vstack([_path(20.0, 90.0), np.full((5, 2), np.nan)])
    assert nav_satnav.alpamayo_text(nav_satnav.instruction(padded)) == "Turn left in 20m"
    assert nav_satnav.alpamayo_text(nav_satnav.instruction(np.full((5, 2), np.nan))) is None


def test_copy_matches_the_teacher_satnav_when_it_is_available():
    import importlib.util
    import pathlib

    src = pathlib.Path(
        "/mnt/efs/users/rod/repos/AgenticDriver-teacher/sim/clients/claude_driver/satnav.py"
    )
    if not src.exists():
        pytest.skip("AgenticDriver checkout not on this machine")
    spec = importlib.util.spec_from_file_location("teacher_satnav", src)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = _path(float(rng.uniform(0, 70)), float(rng.uniform(-200, 200)))
        a, b = orig.instruction(p), nav_satnav.instruction(p)
        assert (a is None) == (b is None)
        if a is not None:
            assert {k: a[k] for k in ("maneuver", "distance_m", "text")} == {
                k: b[k] for k in ("maneuver", "distance_m", "text")
            }
