"""Seat-to-person matching and temporal occupancy utilities.

The module intentionally contains no video loop or model loading so the
matching behavior can be tested with synthetic geometry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot, sqrt
from typing import Deque, Iterable, Optional

import cv2
import numpy as np


UNKNOWN = "UNKNOWN"
OCCUPIED = "OCCUPIED"
AVAILABLE = "AVAILABLE"


@dataclass(frozen=True)
class MatchingConfig:
    """Tunable thresholds and weights for seat matching."""

    min_match_score: float = 0.50
    ambiguity_score: float = 0.35
    reliable_detection_conf: float = 0.25
    lower_body_proximity_weight: float = 0.45
    lower_body_inside_weight: float = 0.25
    center_inside_weight: float = 0.10
    anchor_weight: float = 0.15
    confidence_weight: float = 0.05
    proximity_radius_factor: float = 0.75
    anchor_radius_factor: float = 1.25
    history_size: int = 8
    min_observed_frames: int = 3
    occupied_threshold: float = 0.60
    available_threshold: float = 0.20


@dataclass(frozen=True)
class PersonDetection:
    """One person detection in image coordinates."""

    index: int
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def lower_body(self) -> tuple[float, float]:
        """A stable lower-torso reference, not the feet at y2."""

        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y1 + 0.65 * (y2 - y1))


@dataclass
class SeatRegion:
    """A manually mapped seat polygon and optional physical sitting anchor.

    ``anchor`` is optional for backwards compatibility with the existing
    seats.json. If configured, it should point to the expected hip/torso
    location of a seated person, not necessarily the polygon centroid.
    """

    id: str
    polygon: np.ndarray
    anchor: Optional[tuple[float, float]] = None
    area: float = field(init=False)
    scale: float = field(init=False)
    centroid: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        polygon = np.asarray(self.polygon, dtype=np.int32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError(f"Seat {self.id} must have at least 3 (x, y) points")
        self.polygon = polygon
        self.area = abs(float(cv2.contourArea(polygon)))
        x, y, width, height = cv2.boundingRect(polygon)
        self.scale = max(float(width), float(height), sqrt(max(self.area, 1.0)), 1.0)
        moments = cv2.moments(polygon)
        if moments["m00"]:
            self.centroid = (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            )
        else:
            self.centroid = (float(x + width / 2), float(y + height / 2))

    @classmethod
    def from_json(cls, item: dict) -> "SeatRegion":
        anchor = item.get("anchor")
        return cls(
            id=str(item["id"]),
            polygon=np.asarray(item["polygon"], dtype=np.int32),
            anchor=tuple(map(float, anchor)) if anchor is not None else None,
        )


@dataclass(frozen=True)
class SeatMatch:
    person_index: int
    seat_id: str
    score: float


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _inside_score(point: tuple[float, float], seat: SeatRegion) -> tuple[float, float]:
    """Return (inside flag, signed distance) for a point and polygon."""

    signed_distance = float(cv2.pointPolygonTest(seat.polygon, point, True))
    return (1.0 if signed_distance >= 0.0 else 0.0, signed_distance)


def calculate_occupancy_score(
    person: PersonDetection,
    seat: SeatRegion,
    config: MatchingConfig = MatchingConfig(),
) -> float:
    """Return normalized [0, 1] compatibility for one person-seat pair.

    The score uses the lower-body point as the primary spatial evidence. A
    point just outside a polygon receives a proximity score, but not the
    stronger containment score. This avoids treating every nearby standing
    person as seated. An optional seat anchor adds a second spatial cue.
    """

    lower_inside, lower_signed_distance = _inside_score(person.lower_body, seat)
    center_inside, _ = _inside_score(person.center, seat)

    proximity_radius = max(1.0, seat.scale * config.proximity_radius_factor)
    if lower_signed_distance >= 0.0:
        lower_proximity = 1.0
    else:
        lower_proximity = _clamp(1.0 + lower_signed_distance / proximity_radius)

    components = [
        (config.lower_body_proximity_weight, lower_proximity),
        (config.lower_body_inside_weight, lower_inside),
        (config.center_inside_weight, center_inside),
        (config.confidence_weight, _clamp(float(person.confidence))),
    ]

    if seat.anchor is not None:
        anchor_radius = max(1.0, seat.scale * config.anchor_radius_factor)
        anchor_distance = hypot(
            person.lower_body[0] - seat.anchor[0],
            person.lower_body[1] - seat.anchor[1],
        )
        anchor_score = _clamp(1.0 - anchor_distance / anchor_radius)
        components.insert(3, (config.anchor_weight, anchor_score))

    total_weight = sum(weight for weight, _value in components)
    if total_weight <= 0.0:
        return 0.0
    return _clamp(sum(weight * value for weight, value in components) / total_weight)


def generate_candidate_matches(
    persons: Iterable[PersonDetection],
    seats: Iterable[SeatRegion],
    config: MatchingConfig = MatchingConfig(),
) -> list[SeatMatch]:
    """Generate plausible pairs, including ambiguous pairs for UNKNOWN logic."""

    candidates: list[SeatMatch] = []
    for person in persons:
        for seat in seats:
            score = calculate_occupancy_score(person, seat, config)
            if score >= config.ambiguity_score:
                candidates.append(SeatMatch(person.index, seat.id, score))
    return candidates


def _hungarian_maximize(weights: list[list[float]]) -> list[int]:
    """Return the maximum-weight column assignment for each row.

    This is the O(n^2 m) Hungarian algorithm. The caller adds dummy columns,
    so an assignment can remain unmatched with weight zero.
    """

    if not weights:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if column_count < row_count:
        raise ValueError("Hungarian assignment requires at least as many columns as rows")

    max_weight = max(max(row) for row in weights)
    costs = [[max_weight - value for value in row] for row in weights]
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        p[0] = row
        column_0 = 0
        min_values = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)

        while True:
            used[column_0] = True
            row_0 = p[column_0]
            delta = float("inf")
            next_column = 0
            for column in range(1, column_count + 1):
                if not used[column]:
                    current = costs[row_0 - 1][column - 1] - u[row_0] - v[column]
                    if current < min_values[column]:
                        min_values[column] = current
                        way[column] = column_0
                    if min_values[column] < delta:
                        delta = min_values[column]
                        next_column = column

            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column_0 = next_column
            if p[column_0] == 0:
                break

        while True:
            previous_column = way[column_0]
            p[column_0] = p[previous_column]
            column_0 = previous_column
            if column_0 == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1
    return assignment


def match_persons_to_seats(
    persons: list[PersonDetection],
    seats: list[SeatRegion],
    config: MatchingConfig = MatchingConfig(),
) -> tuple[dict[int, SeatMatch], list[SeatMatch]]:
    """Globally match persons to seats with one-to-one constraints.

    Returns ``person_index -> SeatMatch`` plus all candidate pairs above the
    ambiguity threshold. Dummy columns allow a person to remain unmatched.
    """

    candidates = generate_candidate_matches(persons, seats, config)
    if not persons or not seats:
        return {}, candidates

    seat_index = {seat.id: index for index, seat in enumerate(seats)}
    candidate_scores = {
        (candidate.person_index, candidate.seat_id): candidate.score
        for candidate in candidates
    }
    # Dummy columns represent unmatched persons and have zero score.
    weights = [[0.0 for _ in range(len(seats) + len(persons))] for _ in persons]
    for row, person in enumerate(persons):
        for seat in seats:
            score = candidate_scores.get((person.index, seat.id), 0.0)
            if score >= config.min_match_score:
                weights[row][seat_index[seat.id]] = score

    assignment = _hungarian_maximize(weights)
    matches: dict[int, SeatMatch] = {}
    for row, column in enumerate(assignment):
        if 0 <= column < len(seats):
            person = persons[row]
            seat = seats[column]
            score = weights[row][column]
            if score >= config.min_match_score:
                matches[person.index] = SeatMatch(person.index, seat.id, score)
    return matches, candidates


def update_seat_history(
    history: dict[str, Deque[Optional[int]]],
    last_status: dict[str, str],
    frame_evidence: dict[str, Optional[bool]],
    config: MatchingConfig = MatchingConfig(),
) -> dict[str, str]:
    """Update temporal state without treating missing detections as empty.

    ``True`` means occupied evidence, ``False`` means available evidence, and
    ``None`` means insufficient/ambiguous evidence for this frame.
    """

    statuses: dict[str, str] = {}
    for seat_id, seat_history in history.items():
        evidence = frame_evidence.get(seat_id)
        seat_history.append(None if evidence is None else int(evidence))
        observed = [value for value in seat_history if value is not None]

        if len(seat_history) < config.min_observed_frames or len(observed) < config.min_observed_frames:
            status = (
                last_status.get(seat_id, UNKNOWN)
                if last_status.get(seat_id, UNKNOWN) != UNKNOWN and observed
                else UNKNOWN
            )
        else:
            occupied_ratio = sum(observed) / len(observed)
            if occupied_ratio >= config.occupied_threshold:
                status = OCCUPIED
            elif occupied_ratio <= config.available_threshold:
                status = AVAILABLE
            else:
                status = last_status.get(seat_id, UNKNOWN)

        last_status[seat_id] = status
        statuses[seat_id] = status
    return statuses
