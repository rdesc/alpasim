# SPDX-License-Identifier: Apache-2.0
"""The Claude teacher's satnav decision, worded the way Alpamayo was trained.

``instruction`` is AgenticDriver's ``sim/clients/claude_driver/satnav.py`` (rod/alpasim-teacher ec81e33,
2026-11-12), copied unchanged except that a roundabout also returns its ``net_deg`` (so the exit side can be
worded), since the driver container cannot import that repo. It
reads the recorded drive ahead of the car (x forward, y left, metres from the car's place on the drive) and
decides: straight, turn left/right (>= 25 deg), bear left/right (10-25 deg) or a roundabout, at a distance
bucketed to 0/10/20/30/50/100 m. Here its input is the served route with route_generator_type RECORDED and
route_start_offset_m 0, which is the recorded drive from the car's projection onto it, 80 m ahead (the
teacher reads 100 m).

``alpamayo_text`` words that decision in the phrasing Alpamayo was trained on ("Turn left in 20m",
"Continue straight"; see nav_text.py). Alpamayo has no phrase for a gentle bend or a roundabout exit, so
``bear`` maps by ``bear_as`` and a roundabout by the side of its exit.
"""

from __future__ import annotations

import numpy as np

START_DEG = 8.0
TURN_DEG = 25.0  # 'turn' at or above, 'bear' between BEAR_DEG and this
BEAR_DEG = 10.0
LOOKAHEAD_M = 100.0
DISTANCES = (0, 10, 20, 30, 50, 100)
# Alpamayo's nav labels start a few metres out (nav_text.route_to_nav_text: min_lookahead_m 5).
MIN_REPORTED_M = 5


def _bucket(s):
    if s < 5:
        return 0
    return min(DISTANCES[1:], key=lambda d: abs(d - s))


def _resample(path, step=1.0):
    p = np.asarray(path, float)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        return None, None
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 5:
        return None, None
    g = np.arange(0.0, min(s[-1], 200.0), step)
    return np.stack([np.interp(g, s, p[:, 0]), np.interp(g, s, p[:, 1])], 1), g


def _round_words(net):
    """How far round a right-hand-traffic roundabout the exit is, from the net heading change entry -> exit (+ left):
    -90 a quarter (the first exit of four), 0 half way (straight on), +90 three-quarters, +180 all the way (back)."""
    f = (net + 180.0) / 360.0
    if f < 0.375:
        return 'about a quarter of the way round (to the right)'
    if f < 0.625:
        return 'about half way round (straight on)'
    if f < 0.875:
        return 'about three-quarters of the way round (to the left)'
    return 'nearly all the way round (back the way you came)'


def instruction(path_ahead):
    """{'maneuver', 'distance_m', 'text'} from the recorded path ahead (N x 2, x forward, y left), or None."""
    p, s = _resample(path_ahead)
    if p is None:
        return None
    h = np.degrees(np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0]))))
    d = h - h[0]
    s = s[1:]
    look = s <= LOOKAHEAD_M
    start = np.flatnonzero(look & (np.abs(d) >= START_DEG))
    # roundabout: a right dip on entry, then a left loop of >= 90 deg beyond it (the exit may lie past the recording)
    k_r = np.flatnonzero(look & (d <= -15))
    if len(k_r):
        i0 = k_r[0]
        tail = d[i0:min(len(d), i0 + 120)]
        lo = int(np.argmin(tail[:40])) if len(tail) else 0
        if len(tail) and np.max(tail[lo:]) - tail[lo] >= 90:
            net = float(tail[-1])
            where = _bucket(float(s[start[0]]) if len(start) else float(s[i0]))
            lead = 'Now' if where == 0 else f'In about {where} m'
            text = f"{lead}, enter the roundabout, keep right of the island and leave it {_round_words(net)}."
            return {'maneuver': 'roundabout', 'distance_m': where, 'text': text, 'net_deg': net}
    if len(start):
        i = start[0]
        j = min(len(d) - 1, int(np.searchsorted(s, s[i] + 40.0)))
        turn = float(d[j]) if abs(d[j]) >= abs(d[i]) else float(d[i])
        where = _bucket(float(s[i]))
        side = 'left' if turn > 0 else 'right'
        lead = 'Now' if where == 0 else f'In about {where} m'
        if abs(turn) >= TURN_DEG:
            return {'maneuver': f'turn {side}', 'distance_m': where, 'text': f'{lead}, the route turns {side}.'}
        if abs(turn) >= BEAR_DEG:
            return {'maneuver': f'bear {side}', 'distance_m': where, 'text': f'{lead}, the route bears {side}.'}
    return {'maneuver': 'straight', 'distance_m': 0, 'text': 'Continue straight along the road.'}


def alpamayo_text(instr: dict | None, bear_as: str = "turn") -> str | None:
    """The satnav decision in Alpamayo's trained phrasing, or None (no usable route: run unconditioned).

    bear_as: "turn" words a bear (10-25 deg: a gentle bend or a fork) as a turn to that side, "straight" as
    "Continue straight".
    """
    if instr is None:
        return None
    m = instr['maneuver']
    dist = max(MIN_REPORTED_M, int(instr['distance_m']))
    if m == 'straight':
        return "Continue straight"
    if m == 'roundabout':
        f = (instr.get('net_deg', 0.0) + 180.0) / 360.0
        if f < 0.375:
            return f"Turn right in {dist}m"
        if f < 0.625:
            return "Continue straight"
        return f"Turn left in {dist}m"
    kind, side = m.split()
    if kind == 'bear' and bear_as == 'straight':
        return "Continue straight"
    return f"Turn {side} in {dist}m"
