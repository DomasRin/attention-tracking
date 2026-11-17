from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from eyetrax import GazeEstimator, run_9_point_calibration
from screeninfo import get_monitors


WINDOW_NAME = "Attention Tracker"
ZONE_NAMES = ("zone_1", "zone_2", "zone_3", "zone_4")
ZONE_COLORS = (
    (90, 0, 0),
    (0, 90, 0),
    (0, 0, 90),
    (70, 70, 70),
)


def screen_size() -> tuple[int, int]:
    monitors = get_monitors()
    if not monitors:
        raise RuntimeError("No monitor found")
    primary = next(
        (monitor for monitor in monitors if getattr(monitor, "is_primary", False)),
        monitors[0],
    )
    return primary.width, primary.height


# Four equal zones for now, this should be change in the future
def zone_at(x: float, y: float, width: int, height: int) -> str | None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    column = 1 if x >= width / 2 else 0
    row = 1 if y >= height / 2 else 0
    return ZONE_NAMES[row * 2 + column]


def zone_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    middle_x = width // 2
    middle_y = height // 2
    canvas[:middle_y, :middle_x] = ZONE_COLORS[0]
    canvas[:middle_y, middle_x:] = ZONE_COLORS[1]
    canvas[middle_y:, :middle_x] = ZONE_COLORS[2]
    canvas[middle_y:, middle_x:] = ZONE_COLORS[3]
    cv2.line(canvas, (middle_x, 0), (middle_x, height), (255, 255, 255), 2)
    cv2.line(canvas, (0, middle_y), (width, middle_y), (255, 255, 255), 2)
    return canvas


def export_csv(output: Path, dwell: dict[str, float]) -> None:
    total = sum(dwell.values())
    winner = max(dwell, key=dwell.get) if total > 0 else None
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("zone", "dwell_seconds", "percentage", "most_looked_at"),
        )
        writer.writeheader()
        for zone in ZONE_NAMES:
            seconds = dwell[zone]
            writer.writerow(
                {
                    "zone": zone,
                    "dwell_seconds": f"{seconds:.3f}",
                    "percentage": f"{seconds / total * 100:.2f}" if total else "0.00",
                    "most_looked_at": zone == winner,
                }
            )


def track(camera_index: int) -> dict[str, float]:
    width, height = screen_size()
    dwell = {zone: 0.0 for zone in ZONE_NAMES}
    estimator = GazeEstimator()
    
    # also there are  5 point/lissajous calibrations
    run_9_point_calibration(estimator, camera_index=camera_index)
    if estimator.model is None:
        estimator.face_mesh.close()
        return dwell

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        estimator.face_mesh.close()
        raise RuntimeError(f"Could not open camera {camera_index}")

    background = zone_canvas(width, height)
    smoothed_point: np.ndarray | None = None
    previous_time: float | None = None
    previous_zone: str | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ok, frame = camera.read()
            now = time.perf_counter()
            canvas = background.copy()

            if ok:
                features, blink = estimator.extract_features(frame)
            else:
                features, blink = None, False

            if features is not None and not blink:
                point = estimator.predict(np.array([features]))[0]
                if smoothed_point is None:
                    smoothed_point = point.astype(float)
                else:
                    smoothed_point = 0.25 * point + 0.75 * smoothed_point

                x, y = smoothed_point
                current_zone = zone_at(x, y, width, height)

                if previous_time is not None and previous_zone is not None:
                    elapsed = now - previous_time
                    if elapsed <= 0.25:
                        dwell[previous_zone] += elapsed

                previous_time = now
                previous_zone = current_zone

                if current_zone is not None:
                    cv2.circle(canvas, (int(x), int(y)), 14, (0, 255, 0), -1)
            else:
                smoothed_point = None
                previous_time = None
                previous_zone = None

            cv2.imshow(WINDOW_NAME, canvas)
            if cv2.waitKey(1) == 27:
                break
    finally:
        camera.release()
        estimator.face_mesh.close()
        cv2.destroyAllWindows()

    return dwell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("attention_results.csv"))
    args = parser.parse_args()

    dwell = track(args.camera)
    export_csv(args.output, dwell)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
