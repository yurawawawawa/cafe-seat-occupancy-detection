import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "CCTV FOOTAGE.mp4"
SEATS_PATH = BASE_DIR / "seats.json"

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Gagal membuka video: {VIDEO_PATH}")

ret, first_frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Gagal membaca frame pertama video.")

seats = []
current_points = []


def draw_overlay(image):
    output = image.copy()

    for seat in seats:
        polygon = seat["polygon"]
        cv2.polylines(
            output,
            [np.array(polygon, dtype=np.int32)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )
        center_x = int(sum(point[0] for point in polygon) / len(polygon))
        center_y = int(sum(point[1] for point in polygon) / len(polygon))
        cv2.putText(
            output,
            seat["id"],
            (center_x - 30, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if current_points:
        for point in current_points:
            cv2.circle(output, tuple(point), 5, (0, 0, 255), -1)
        if len(current_points) > 1:
            cv2.polylines(
                output,
                [np.array(current_points, dtype=np.int32)],
                isClosed=False,
                color=(0, 0, 255),
                thickness=2,
            )

    cv2.putText(
        output,
        "Klik 4+ titik per kursi | ENTER: simpan | U: undo | R: reset | Q: selesai",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return output


def on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])


window_name = "Seat Mapping"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, on_mouse)

while True:
    cv2.imshow(window_name, draw_overlay(first_frame))
    key = cv2.waitKey(30) & 0xFF

    if key == ord("u") and current_points:
        current_points.pop()
    elif key == ord("r"):
        current_points.clear()
    elif key in (13, 32):
        if len(current_points) >= 3:
            seats.append(
                {
                    "id": f"seat_{len(seats) + 1:02d}",
                    "polygon": current_points.copy(),
                }
            )
            current_points.clear()
    elif key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()

if not seats:
    print("Tidak ada seat yang disimpan.")
else:
    SEATS_PATH.write_text(json.dumps(seats, indent=2), encoding="utf-8")
    print(f"Tersimpan {len(seats)} seat ke {SEATS_PATH.name}.")
