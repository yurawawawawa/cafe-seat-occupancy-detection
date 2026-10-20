from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from occupancy import (
    AVAILABLE,
    OCCUPIED,
    UNKNOWN,
    MatchingConfig,
    PersonDetection,
    SeatMatch,
    SeatRegion,
    match_persons_to_seats,
    update_seat_history,
)

try:
    import torch

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "CCTV FOOTAGE.mp4"
SEATS_PATH = BASE_DIR / "seats.json"
MODEL_PATH = BASE_DIR / "yolo11n.pt"

DETECTION_CONFIDENCE = 0.20
DEBUG_MATCHING = False
CONFIG = MatchingConfig(
    min_match_score=0.50,
    ambiguity_score=0.35,
    reliable_detection_conf=0.25,
)

COLORS = {
    OCCUPIED: (0, 0, 255),
    AVAILABLE: (0, 180, 0),
    UNKNOWN: (0, 165, 255),
}
PERSON_COLOR = (255, 0, 0)  # Blue in BGR.


def load_seats(path: Path) -> list[SeatRegion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    seats = [SeatRegion.from_json(item) for item in data]
    ids = [seat.id for seat in seats]
    if len(ids) != len(set(ids)):
        raise ValueError("seats.json contains duplicate seat IDs")
    return seats


def draw_occupancy_overlay(
    frame,
    persons: list[PersonDetection],
    seats: list[SeatRegion],
    statuses: dict[str, str],
    matches: dict[int, SeatMatch],
    debug: bool = False,
):
    """Draw seat status, blue person boxes, and optional associations."""

    output = frame.copy()
    match_by_seat = {match.seat_id: match for match in matches.values()}

    for person in persons:
        x1, y1, x2, y2 = map(int, person.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), PERSON_COLOR, 2)
        if debug:
            match = matches.get(person.index)
            label = f"person_{person.index + 1} {person.confidence:.2f}"
            if match:
                label += f" -> {match.seat_id} {match.score:.2f}"
            cv2.putText(
                output,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                PERSON_COLOR,
                1,
            )

    for seat in seats:
        status = statuses.get(seat.id, UNKNOWN)
        color = COLORS[status]
        cv2.polylines(output, [seat.polygon], True, color, 3)
        label_x, label_y = map(int, seat.centroid)
        cv2.putText(
            output,
            f"{seat.id}: {status}",
            (label_x - 50, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            2,
        )

        if debug and seat.id in match_by_seat:
            match = match_by_seat[seat.id]
            person = next(item for item in persons if item.index == match.person_index)
            target = seat.anchor if seat.anchor is not None else seat.centroid
            cv2.line(
                output,
                tuple(map(int, person.lower_body)),
                tuple(map(int, target)),
                (255, 255, 0),
                2,
            )

    counts = {
        OCCUPIED: sum(status == OCCUPIED for status in statuses.values()),
        AVAILABLE: sum(status == AVAILABLE for status in statuses.values()),
        UNKNOWN: sum(status == UNKNOWN for status in statuses.values()),
    }
    total = len(seats)
    assert total == counts[OCCUPIED] + counts[AVAILABLE] + counts[UNKNOWN]

    panel_height = 148 if debug else 126
    cv2.rectangle(output, (10, 45), (390, panel_height), (0, 0, 0), -1)
    panel_lines = [
        f"Total Seats: {total}",
        f"Occupied: {counts[OCCUPIED]}",
        f"Available: {counts[AVAILABLE]}",
        f"Unknown: {counts[UNKNOWN]}",
    ]
    for line_index, text in enumerate(panel_lines):
        cv2.putText(
            output,
            text,
            (20, 70 + line_index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
    if debug:
        cv2.putText(
            output,
            "DEBUG ON | D: toggle | Q: quit",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 0),
            1,
        )
    return output


if not SEATS_PATH.exists():
    raise FileNotFoundError(
        "seats.json belum ada. Jalankan terlebih dahulu: "
        ".\\.venv\\Scripts\\python.exe map_seats.py"
    )

seats = load_seats(SEATS_PATH)
model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Gagal membuka video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 12
delay = max(1, int(1000 / fps))
history = {
    seat.id: deque(maxlen=CONFIG.history_size)
    for seat in seats
}
last_status = {seat.id: UNKNOWN for seat in seats}

print(
    f"Video CCTV aktif ({fps:.1f} FPS), seats={len(seats)}, "
    f"device={DEVICE}. D: debug, Q: keluar."
)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model.predict(
        frame,
        classes=[0],
        conf=DETECTION_CONFIDENCE,
        imgsz=960,
        iou=0.70,
        max_det=300,
        augment=True,
        device=DEVICE,
        verbose=False,
    )

    persons: list[PersonDetection] = []
    for index, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        persons.append(PersonDetection(index, (x1, y1, x2, y2), confidence))

    matches, candidates = match_persons_to_seats(persons, seats, CONFIG)
    matched_seats = {match.seat_id for match in matches.values()}
    candidate_by_seat: dict[str, list[float]] = {seat.id: [] for seat in seats}
    for candidate in candidates:
        candidate_by_seat[candidate.seat_id].append(candidate.score)

    reliable_people = any(
        person.confidence >= CONFIG.reliable_detection_conf for person in persons
    )
    frame_evidence: dict[str, bool | None] = {}
    for seat in seats:
        if seat.id in matched_seats:
            frame_evidence[seat.id] = True
        elif not reliable_people:
            # No reliable scene observation: do not convert missing detection to empty.
            frame_evidence[seat.id] = None
        elif any(score >= CONFIG.ambiguity_score for score in candidate_by_seat[seat.id]):
            # A nearby/overlapping person exists, but matching was not confident enough.
            frame_evidence[seat.id] = None
        else:
            frame_evidence[seat.id] = False

    statuses = update_seat_history(history, last_status, frame_evidence, CONFIG)
    annotated = draw_occupancy_overlay(frame, persons, seats, statuses, matches, DEBUG_MATCHING)
    cv2.imshow("Cafe Fixed Seat Occupancy", annotated)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break
    if key == ord("d"):
        DEBUG_MATCHING = not DEBUG_MATCHING

cap.release()
cv2.destroyAllWindows()
