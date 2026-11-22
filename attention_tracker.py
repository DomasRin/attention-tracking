
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from eyetrax import GazeEstimator
from screeninfo import get_monitors

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtWidgets import QApplication, QWidget, QPushButton

from pynput import keyboard


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ZONE_NAMES = ("zone_1", "zone_2", "zone_3", "zone_4")

ZONE_COLORS = (
    (90, 0, 0),
    (0, 90, 0),
    (0, 0, 90),
    (70, 70, 70),
)

OVERLAY_OPACITY = 0.20

GAZE_RADIUS = 12
GAZE_ALPHA = 110

MEASUREMENT_SMOOTHING = 0.25
DISPLAY_SMOOTHING = 0.10

MAX_SAMPLE_GAP = 0.25

# Calibration settings.
GRID_ROWS = 4
GRID_COLS = 4

LISSAJOUS_DURATION = 8.0

GRID_SETTLE_TIME = 0.25
GRID_SAMPLE_TIME = 0.75

TARGET_MOVE_TIME = 0.35

VALIDATION_SETTLE_TIME = 0.25
VALIDATION_SAMPLE_TIME = 0.85

CALIBRATION_MARGIN = 0.08

# Recenter settings.
RECENTER_SAMPLE_TIME = 1.8
RECENTER_SETTLE_TIME = 0.4

# How quickly the correction moves towards its new value.
# Smaller values create a slower, smoother adjustment.
RECENTER_ADJUSTMENT_SPEED = 0.08

# Minimum valid predictions required for a recenter.
RECENTER_MIN_SAMPLES = 8


# ---------------------------------------------------------
# SCREEN
# ---------------------------------------------------------

def screen_size() -> tuple[int, int, int, int]:

    monitors = get_monitors()

    if not monitors:
        raise RuntimeError("No monitor found")

    primary = next(
        (
            monitor
            for monitor in monitors
            if getattr(monitor, "is_primary", False)
        ),
        monitors[0],
    )

    return (
        primary.width,
        primary.height,
        getattr(primary, "x", 0),
        getattr(primary, "y", 0),
    )


# ---------------------------------------------------------
# ZONE CLASSIFICATION
# ---------------------------------------------------------

def zone_at(x, y, width, height):

    if x < 0 or y < 0 or x >= width or y >= height:
        return None

    column = 1 if x >= width / 2 else 0
    row = 1 if y >= height / 2 else 0

    return ZONE_NAMES[row * 2 + column]


# ---------------------------------------------------------
# CALIBRATION WINDOW
# ---------------------------------------------------------

class CalibrationWindow(QWidget):

    def __init__(self, width, height):

        super().__init__()

        self.screen_width = width
        self.screen_height = height

        self.target = None
        self.status = "Preparing calibration..."
        self.cancelled = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
        )

        self.setStyleSheet("background-color: #141414;")

    def set_target(self, x, y, status=None):

        self.target = (int(x), int(y))

        if status is not None:
            self.status = status

        self.update()

    def keyPressEvent(self, event):

        if event.key() == Qt.Key.Key_Escape:
            self.cancelled = True

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        painter.fillRect(
            self.rect(),
            QColor(20, 20, 24),
        )

        painter.setPen(QColor(230, 230, 230))
        painter.setFont(QFont("Arial", 18))

        painter.drawText(
            QRect(
                0,
                35,
                self.screen_width,
                40,
            ),
            Qt.AlignmentFlag.AlignCenter,
            self.status,
        )

        painter.setFont(QFont("Arial", 11))
        painter.setPen(QColor(150, 150, 150))

        painter.drawText(
            QRect(
                0,
                self.screen_height - 55,
                self.screen_width,
                30,
            ),
            Qt.AlignmentFlag.AlignCenter,
            "Follow the target with your eyes. Keep your head relatively still. ESC to cancel.",
        )

        if self.target is not None:

            x, y = self.target

            painter.setPen(
                QPen(QColor(0, 255, 120, 100), 3)
            )

            painter.setBrush(
                QBrush(QColor(0, 255, 120, 35))
            )

            painter.drawEllipse(
                x - 24,
                y - 24,
                48,
                48,
            )

            painter.setPen(
                QPen(QColor(255, 255, 255), 2)
            )

            painter.setBrush(
                QBrush(QColor(0, 255, 120))
            )

            painter.drawEllipse(
                x - 10,
                y - 10,
                20,
                20,
            )

        painter.end()


# ---------------------------------------------------------
# CAMERA MANAGEMENT
# ---------------------------------------------------------

def open_cameras(index_1, index_2):

    camera_1 = cv2.VideoCapture(index_1)
    camera_2 = cv2.VideoCapture(index_2)

    if not camera_1.isOpened():

        camera_1.release()
        camera_2.release()

        raise RuntimeError(
            f"Could not open camera {index_1}"
        )

    if not camera_2.isOpened():

        camera_1.release()
        camera_2.release()

        raise RuntimeError(
            f"Could not open camera {index_2}"
        )

    camera_1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera_2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera_1, camera_2


def read_frames(camera_1, camera_2):

    ok_1, frame_1 = camera_1.read()
    ok_2, frame_2 = camera_2.read()

    if not ok_1:
        frame_1 = None

    if not ok_2:
        frame_2 = None

    return frame_1, frame_2


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_valid_features(estimator, frame):

    if frame is None:
        return None

    features, blink = estimator.extract_features(frame)

    if features is None or blink:
        return None

    features = np.asarray(features, dtype=float)

    if not np.all(np.isfinite(features)):
        return None

    return features


def predict_from_features(estimator, features):

    if features is None:
        return None

    prediction = estimator.predict(
        np.asarray([features])
    )[0]

    prediction = np.asarray(prediction, dtype=float)

    if not np.all(np.isfinite(prediction)):
        return None

    return prediction


# ---------------------------------------------------------
# CALIBRATION HELPERS
# ---------------------------------------------------------

def grid_points(width, height):

    xs = np.linspace(
        CALIBRATION_MARGIN,
        1.0 - CALIBRATION_MARGIN,
        GRID_COLS,
    )

    ys = np.linspace(
        CALIBRATION_MARGIN,
        1.0 - CALIBRATION_MARGIN,
        GRID_ROWS,
    )

    points = []

    for row, y in enumerate(ys):

        row_xs = xs if row % 2 == 0 else xs[::-1]

        for x in row_xs:

            points.append(
                (
                    int(x * width),
                    int(y * height),
                )
            )

    return points


def animate_target(
    app,
    window,
    camera_1,
    camera_2,
    start,
    end,
    duration,
):

    start_time = time.perf_counter()

    while True:

        if window.cancelled:
            raise KeyboardInterrupt("Calibration cancelled")

        elapsed = time.perf_counter() - start_time

        progress = min(elapsed / duration, 1.0)
        progress = progress * progress * (3 - 2 * progress)

        x = start[0] + (end[0] - start[0]) * progress
        y = start[1] + (end[1] - start[1]) * progress

        window.set_target(x, y)
        app.processEvents()

        read_frames(camera_1, camera_2)

        if elapsed >= duration:
            break


def collect_calibration_sample(
    estimator_1,
    estimator_2,
    frame_1,
    frame_2,
    target,
    features_1,
    targets_1,
    features_2,
    targets_2,
):

    feature_1 = extract_valid_features(
        estimator_1,
        frame_1,
    )

    feature_2 = extract_valid_features(
        estimator_2,
        frame_2,
    )

    if feature_1 is not None:
        features_1.append(feature_1)
        targets_1.append(target)

    if feature_2 is not None:
        features_2.append(feature_2)
        targets_2.append(target)


# ---------------------------------------------------------
# IMPROVED TWO-CAMERA CALIBRATION
# ---------------------------------------------------------

def calibrate_cameras(
    app,
    window,
    camera_1,
    camera_2,
    estimator_1,
    estimator_2,
    width,
    height,
):

    features_1 = []
    targets_1 = []

    features_2 = []
    targets_2 = []

    print("\nStarting calibration...")
    print("Follow the moving green target.")

    # Phase 1: Lissajous movement.

    print("\nPhase 1: Moving-target calibration")

    start_time = time.perf_counter()

    while True:

        if window.cancelled:
            raise KeyboardInterrupt("Calibration cancelled")

        elapsed = time.perf_counter() - start_time

        if elapsed >= LISSAJOUS_DURATION:
            break

        progress = elapsed / LISSAJOUS_DURATION
        angle = progress * 2 * np.pi

        x = width / 2 + width * 0.38 * np.sin(3 * angle)
        y = height / 2 + height * 0.38 * np.sin(2 * angle)

        target = (int(x), int(y))

        window.set_target(
            x,
            y,
            "Phase 1/2: Follow the moving target",
        )

        app.processEvents()

        frame_1, frame_2 = read_frames(
            camera_1,
            camera_2,
        )

        collect_calibration_sample(
            estimator_1,
            estimator_2,
            frame_1,
            frame_2,
            target,
            features_1,
            targets_1,
            features_2,
            targets_2,
        )

    # Phase 2: Dense stationary grid.

    print("\nPhase 2: Dense grid calibration")

    points = grid_points(width, height)

    previous_target = (
        width // 2,
        height // 2,
    )

    for index, target in enumerate(points):

        if window.cancelled:
            raise KeyboardInterrupt("Calibration cancelled")

        print(
            f"Calibration point {index + 1}/{len(points)}"
        )

        animate_target(
            app,
            window,
            camera_1,
            camera_2,
            previous_target,
            target,
            TARGET_MOVE_TIME,
        )

        window.set_target(
            *target,
            status=(
                f"Phase 2/2: Grid calibration "
                f"({index + 1}/{len(points)})"
            ),
        )

        settle_start = time.perf_counter()

        while (
            time.perf_counter() - settle_start
            < GRID_SETTLE_TIME
        ):

            if window.cancelled:
                raise KeyboardInterrupt("Calibration cancelled")

            read_frames(camera_1, camera_2)
            app.processEvents()

        sample_start = time.perf_counter()

        while (
            time.perf_counter() - sample_start
            < GRID_SAMPLE_TIME
        ):

            if window.cancelled:
                raise KeyboardInterrupt("Calibration cancelled")

            frame_1, frame_2 = read_frames(
                camera_1,
                camera_2,
            )

            collect_calibration_sample(
                estimator_1,
                estimator_2,
                frame_1,
                frame_2,
                target,
                features_1,
                targets_1,
                features_2,
                targets_2,
            )

            app.processEvents()

        previous_target = target

    if len(features_1) < 20:
        raise RuntimeError(
            f"Camera 1 collected only {len(features_1)} valid samples."
        )

    if len(features_2) < 20:
        raise RuntimeError(
            f"Camera 2 collected only {len(features_2)} valid samples."
        )

    print("\nTraining camera 1 model...")

    estimator_1.train(
        np.asarray(features_1),
        np.asarray(targets_1),
    )

    print("Training camera 2 model...")

    estimator_2.train(
        np.asarray(features_2),
        np.asarray(targets_2),
    )

    print("\nCalibration complete.")
    print(f"Camera 1 samples: {len(features_1)}")
    print(f"Camera 2 samples: {len(features_2)}")

    return (
        len(features_1),
        len(features_2),
    )


# ---------------------------------------------------------
# CALIBRATION VALIDATION
# ---------------------------------------------------------

def validation_points(width, height):

    positions = (
        (0.50, 0.50),
        (0.50, 0.17),
        (0.50, 0.83),
        (0.17, 0.50),
        (0.83, 0.50),
    )

    return [
        (int(x * width), int(y * height))
        for x, y in positions
    ]


def validate_cameras(
    app,
    window,
    camera_1,
    camera_2,
    estimator_1,
    estimator_2,
    width,
    height,
):

    print("\nStarting independent camera validation.")

    all_errors = [[], []]

    estimators = (
        estimator_1,
        estimator_2,
    )

    points = validation_points(width, height)

    for index, target in enumerate(points):

        if window.cancelled:
            raise KeyboardInterrupt("Validation cancelled")

        window.set_target(
            *target,
            status=(
                f"Validation: Look at target "
                f"({index + 1}/{len(points)})"
            ),
        )

        app.processEvents()

        predictions = [[], []]

        start = time.perf_counter()

        while (
            time.perf_counter() - start
            < VALIDATION_SETTLE_TIME
        ):

            read_frames(camera_1, camera_2)
            app.processEvents()

        start = time.perf_counter()

        while (
            time.perf_counter() - start
            < VALIDATION_SAMPLE_TIME
        ):

            if window.cancelled:
                raise KeyboardInterrupt("Validation cancelled")

            frames = read_frames(camera_1, camera_2)

            for camera_idx in range(2):

                features = extract_valid_features(
                    estimators[camera_idx],
                    frames[camera_idx],
                )

                if features is None:
                    continue

                prediction = predict_from_features(
                    estimators[camera_idx],
                    features,
                )

                if prediction is not None:
                    predictions[camera_idx].append(
                        prediction
                    )

            app.processEvents()

        for camera_idx in range(2):

            if not predictions[camera_idx]:
                continue

            median_prediction = np.median(
                np.asarray(predictions[camera_idx]),
                axis=0,
            )

            error = np.linalg.norm(
                median_prediction - np.asarray(target)
            )

            all_errors[camera_idx].append(float(error))

    mean_errors = [
        float(np.mean(errors)) if errors else float("inf")
        for errors in all_errors
    ]

    inverse_errors = [
        (
            1.0 / max(error, 1.0)
            if np.isfinite(error)
            else 0.0
        )
        for error in mean_errors
    ]

    total = sum(inverse_errors)

    if total > 0:
        weights = [value / total for value in inverse_errors]
    else:
        weights = [0.5, 0.5]

    print("\nValidation results:")

    for index, error in enumerate(mean_errors):

        error_text = (
            f"{error:.1f} pixels"
            if np.isfinite(error)
            else "Unavailable"
        )

        print(
            f"Camera {index + 1}: "
            f"{error_text}, "
            f"fusion weight: {weights[index]:.3f}"
        )

    return mean_errors, weights


# ---------------------------------------------------------
# CALIBRATION REPORT
# ---------------------------------------------------------

def export_calibration_report(
    output,
    sample_counts,
    errors,
    weights,
):

    report_path = output.with_name(
        output.stem + "_calibration.csv"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                "camera",
                "training_samples",
                "validation_error_pixels",
                "fusion_weight",
            ),
        )

        writer.writeheader()

        for index in range(2):

            writer.writerow(
                {
                    "camera": index + 1,
                    "training_samples": sample_counts[index],
                    "validation_error_pixels": (
                        f"{errors[index]:.2f}"
                        if np.isfinite(errors[index])
                        else "unavailable"
                    ),
                    "fusion_weight": f"{weights[index]:.4f}",
                }
            )

    print(f"\nCalibration report: {report_path.resolve()}")


# ---------------------------------------------------------
# TRANSPARENT DESKTOP OVERLAY
# ---------------------------------------------------------

class GazeOverlay(QWidget):

    def __init__(
        self,
        width,
        height,
        x=0,
        y=0,
        opacity=OVERLAY_OPACITY,
    ):

        super().__init__()

        self.screen_width = width
        self.screen_height = height

        self.opacity = max(
            0.0,
            min(1.0, opacity),
        )

        self.gaze_point = None

        self.setGeometry(
            x,
            y,
            width,
            height,
        )

        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )

        self.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating
        )

        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )

        self.show()

    def update_gaze(self, x=None, y=None):

        if x is None or y is None:
            self.gaze_point = None
        else:
            self.gaze_point = (int(x), int(y))

        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        w = self.screen_width
        h = self.screen_height

        mx = w // 2
        my = h // 2

        alpha = int(255 * self.opacity)

        rectangles = (
            QRect(0, 0, mx, my),
            QRect(mx, 0, w - mx, my),
            QRect(0, my, mx, h - my),
            QRect(mx, my, w - mx, h - my),
        )

        for color, rect in zip(ZONE_COLORS, rectangles):

            r, g, b = color

            painter.fillRect(
                rect,
                QColor(r, g, b, alpha),
            )

        painter.setPen(
            QPen(QColor(255, 255, 255, 180), 2)
        )

        painter.drawLine(mx, 0, mx, h)
        painter.drawLine(0, my, w, my)

        if self.gaze_point is not None:

            x, y = self.gaze_point

            painter.setPen(
                QPen(QColor(0, 255, 100, 100), 2)
            )

            painter.setBrush(
                QBrush(QColor(0, 255, 100, 35))
            )

            painter.drawEllipse(
                x - GAZE_RADIUS - 3,
                y - GAZE_RADIUS - 3,
                (GAZE_RADIUS + 3) * 2,
                (GAZE_RADIUS + 3) * 2,
            )

            painter.setPen(
                QPen(QColor(0, 255, 100, 160), 1)
            )

            painter.setBrush(
                QBrush(QColor(0, 255, 100, GAZE_ALPHA))
            )

            painter.drawEllipse(
                x - GAZE_RADIUS,
                y - GAZE_RADIUS,
                GAZE_RADIUS * 2,
                GAZE_RADIUS * 2,
            )

        painter.end()


# ---------------------------------------------------------
# RECENTER TARGET WINDOW
# ---------------------------------------------------------

class RecenterWindow(QWidget):

    def __init__(
        self,
        width,
        height,
        screen_x,
        screen_y,
    ):

        super().__init__()

        self.screen_width = width
        self.screen_height = height

        self.status = "Look at the centre target and hold your gaze."

        self.setGeometry(
            screen_x,
            screen_y,
            width,
            height,
        )

        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )

        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )

    def set_status(self, status):

        self.status = status
        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        # A subtle dark veil keeps the target visible over other apps.
        painter.fillRect(
            self.rect(),
            QColor(0, 0, 0, 65),
        )

        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 18))

        painter.drawText(
            QRect(
                0,
                35,
                self.screen_width,
                45,
            ),
            Qt.AlignmentFlag.AlignCenter,
            self.status,
        )

        cx = self.screen_width // 2
        cy = self.screen_height // 2

        painter.setPen(
            QPen(QColor(0, 255, 120, 100), 3)
        )

        painter.setBrush(
            QBrush(QColor(0, 255, 120, 35))
        )

        painter.drawEllipse(
            cx - 28,
            cy - 28,
            56,
            56,
        )

        painter.setPen(
            QPen(QColor(255, 255, 255), 2)
        )

        painter.setBrush(
            QBrush(QColor(0, 255, 120))
        )

        painter.drawEllipse(
            cx - 10,
            cy - 10,
            20,
            20,
        )

        painter.end()


# ---------------------------------------------------------
# RECENTER BUTTON
# ---------------------------------------------------------

class RecenterButton(QWidget):

    def __init__(
        self,
        screen_width,
        screen_x,
        screen_y,
        request_event,
    ):

        super().__init__()

        self.request_event = request_event

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )

        self.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating
        )

        self.setFixedSize(112, 38)

        self.setGeometry(
            screen_x + screen_width - 132,
            screen_y + 18,
            112,
            38,
        )

        self.button = QPushButton("Recenter", self)

        self.button.setGeometry(
            0,
            0,
            112,
            38,
        )

        self.button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.button.setStyleSheet("""
            QPushButton {
                background-color: rgba(20, 35, 30, 225);
                color: white;
                border: 1px solid rgba(0, 255, 120, 180);
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: rgba(0, 120, 65, 235);
            }

            QPushButton:pressed {
                background-color: rgba(0, 80, 45, 235);
            }
        """)

        self.button.clicked.connect(
            self.request_recenter
        )

        self.show()

    def request_recenter(self):

        self.request_event.set()


# ---------------------------------------------------------
# TWO-CAMERA PREDICTION FUSION
# ---------------------------------------------------------

def combine_predictions(
    point_1,
    point_2,
    weights,
):

    points = (point_1, point_2)

    valid = [
        index
        for index, point in enumerate(points)
        if point is not None
    ]

    if not valid:
        return None

    active_weights = [
        weights[index]
        for index in valid
    ]

    total = sum(active_weights)

    if total <= 0:
        active_weights = [
            1.0 / len(valid)
            for _ in valid
        ]
    else:
        active_weights = [
            weight / total
            for weight in active_weights
        ]

    result = np.zeros(2, dtype=float)

    for index, weight in zip(valid, active_weights):
        result += points[index] * weight

    return result


# ---------------------------------------------------------
# RECENTERING PROCESS
# ---------------------------------------------------------

def perform_recenter(
    app,
    camera_1,
    camera_2,
    estimator_1,
    estimator_2,
    weights,
    width,
    height,
    screen_x,
    screen_y,
    stop_event,
):

    print("\nRecenter requested.")

    recenter_window = RecenterWindow(
        width,
        height,
        screen_x,
        screen_y,
    )

    recenter_window.show()
    app.processEvents()

    estimators = (
        estimator_1,
        estimator_2,
    )

    samples = []

    try:

        # Initial settling period.
        recenter_window.set_status(
            "Look at the centre target. Keep your head comfortable and steady."
        )

        settle_start = time.perf_counter()

        while (
            time.perf_counter() - settle_start
            < RECENTER_SETTLE_TIME
        ):

            if stop_event.is_set():
                return None

            read_frames(camera_1, camera_2)
            app.processEvents()

        # Collect raw predictions without applying the old offset.
        recenter_window.set_status(
            "Hold your gaze on the target..."
        )

        sample_start = time.perf_counter()

        while (
            time.perf_counter() - sample_start
            < RECENTER_SAMPLE_TIME
        ):

            if stop_event.is_set():
                return None

            frame_1, frame_2 = read_frames(
                camera_1,
                camera_2,
            )

            features_1 = extract_valid_features(
                estimator_1,
                frame_1,
            )

            features_2 = extract_valid_features(
                estimator_2,
                frame_2,
            )

            point_1 = predict_from_features(
                estimator_1,
                features_1,
            )

            point_2 = predict_from_features(
                estimator_2,
                features_2,
            )

            combined = combine_predictions(
                point_1,
                point_2,
                weights,
            )

            if combined is not None:
                samples.append(combined)

            app.processEvents()

        if len(samples) < RECENTER_MIN_SAMPLES:

            recenter_window.set_status(
                "Not enough valid eye samples. Recenter cancelled."
            )

            print(
                f"Recenter cancelled: only {len(samples)} valid samples."
            )

            time.sleep(0.7)

            return None

        # Use the median to reduce outlier influence.
        median_prediction = np.median(
            np.asarray(samples),
            axis=0,
        )

        target = np.asarray(
            [
                width / 2,
                height / 2,
            ],
            dtype=float,
        )

        new_offset = target - median_prediction

        recenter_window.set_status(
            "Recenter complete!"
        )

        print(
            "Recenter offset calculated: "
            f"x={new_offset[0]:.1f}, "
            f"y={new_offset[1]:.1f}"
        )

        time.sleep(0.5)

        return new_offset

    finally:

        recenter_window.close()
        app.processEvents()


# ---------------------------------------------------------
# DWELL-TIME EXPORT
# ---------------------------------------------------------

def export_csv(output, dwell):

    total = sum(dwell.values())

    winner = (
        max(dwell, key=dwell.get)
        if total > 0
        else None
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                "zone",
                "dwell_seconds",
                "percentage",
                "most_looked_at",
            ),
        )

        writer.writeheader()

        for zone in ZONE_NAMES:

            seconds = dwell[zone]

            writer.writerow(
                {
                    "zone": zone,
                    "dwell_seconds": f"{seconds:.3f}",
                    "percentage": (
                        f"{seconds / total * 100:.2f}"
                        if total
                        else "0.00"
                    ),
                    "most_looked_at": zone == winner,
                }
            )


# ---------------------------------------------------------
# MAIN TRACKING LOOP
# ---------------------------------------------------------

def track(
    camera_index_1,
    camera_index_2,
    output,
):

    width, height, screen_x, screen_y = screen_size()

    app = (
        QApplication.instance()
        or QApplication(sys.argv)
    )

    app.setQuitOnLastWindowClosed(False)

    estimator_1 = GazeEstimator()
    estimator_2 = GazeEstimator()

    estimators = (
        estimator_1,
        estimator_2,
    )

    camera_1 = None
    camera_2 = None

    calibration_window = None
    validation_window = None
    overlay = None
    recenter_button = None

    listener = None

    try:

        camera_1, camera_2 = open_cameras(
            camera_index_1,
            camera_index_2,
        )

        # Initial calibration.
        calibration_window = CalibrationWindow(
            width,
            height,
        )

        calibration_window.setGeometry(
            screen_x,
            screen_y,
            width,
            height,
        )

        calibration_window.showFullScreen()
        calibration_window.activateWindow()

        app.processEvents()

        sample_counts = calibrate_cameras(
            app,
            calibration_window,
            camera_1,
            camera_2,
            estimator_1,
            estimator_2,
            width,
            height,
        )

        calibration_window.close()
        calibration_window = None

        # Validation.
        validation_window = CalibrationWindow(
            width,
            height,
        )

        validation_window.setGeometry(
            screen_x,
            screen_y,
            width,
            height,
        )

        validation_window.showFullScreen()
        validation_window.activateWindow()

        app.processEvents()

        errors, weights = validate_cameras(
            app,
            validation_window,
            camera_1,
            camera_2,
            estimator_1,
            estimator_2,
            width,
            height,
        )

        validation_window.close()
        validation_window = None

        export_calibration_report(
            output,
            sample_counts,
            errors,
            weights,
        )

        # Start the tracking overlay.
        overlay = GazeOverlay(
            width,
            height,
            screen_x,
            screen_y,
            OVERLAY_OPACITY,
        )

        dwell = {
            zone: 0.0
            for zone in ZONE_NAMES
        }

        smoothed_measurement = None
        smoothed_display = None

        # Current and desired recenter corrections.
        correction_offset = np.zeros(2, dtype=float)
        desired_offset = np.zeros(2, dtype=float)

        previous_time = None
        previous_zone = None

        stop_event = Event()
        recenter_event = Event()

        # Create the clickable control.
        recenter_button = RecenterButton(
            width,
            screen_x,
            screen_y,
            recenter_event,
        )

        # Global keyboard controls.
        def on_press(key):

            if key == keyboard.Key.esc:
                stop_event.set()
                return False

            # Physical key below Escape:
            # backtick without Shift, tilde with Shift.
            try:
                if key.char in ("`", "~"):
                    recenter_event.set()
            except AttributeError:
                pass

        listener = keyboard.Listener(
            on_press=on_press,
            daemon=True,
        )

        listener.start()

        print("\nAttention tracking started.")
        print("Press ESC to finish.")
        print("Press ` or ~ to recenter.")
        print("You can also click the Recenter button.")

        try:

            while not stop_event.is_set():

                # -----------------------------------------
                # CHECK FOR RECENTER REQUEST
                # -----------------------------------------

                if recenter_event.is_set():

                    recenter_event.clear()

                    # Stop dwell measurement during recenter.
                    previous_time = None
                    previous_zone = None

                    smoothed_measurement = None
                    smoothed_display = None

                    overlay.update_gaze(None)

                    new_offset = perform_recenter(
                        app,
                        camera_1,
                        camera_2,
                        estimator_1,
                        estimator_2,
                        weights,
                        width,
                        height,
                        screen_x,
                        screen_y,
                        stop_event,
                    )

                    if new_offset is not None:

                        desired_offset = new_offset.copy()

                        print(
                            "New recenter correction accepted."
                        )

                    else:

                        print(
                            "Existing recenter correction retained."
                        )

                    # Restart measurement timing cleanly.
                    previous_time = None
                    previous_zone = None

                    continue

                frame_1, frame_2 = read_frames(
                    camera_1,
                    camera_2,
                )

                now = time.perf_counter()

                features_1 = extract_valid_features(
                    estimator_1,
                    frame_1,
                )

                features_2 = extract_valid_features(
                    estimator_2,
                    frame_2,
                )

                point_1 = predict_from_features(
                    estimator_1,
                    features_1,
                )

                point_2 = predict_from_features(
                    estimator_2,
                    features_2,
                )

                combined = combine_predictions(
                    point_1,
                    point_2,
                    weights,
                )

                if combined is not None:

                    # Gradually move the current offset towards
                    # the newly requested recenter correction.
                    correction_offset += (
                        RECENTER_ADJUSTMENT_SPEED
                        * (desired_offset - correction_offset)
                    )

                    corrected = combined + correction_offset

                    # Measurement smoothing.
                    if smoothed_measurement is None:

                        smoothed_measurement = corrected.copy()

                    else:

                        smoothed_measurement = (
                            MEASUREMENT_SMOOTHING * corrected
                            + (1 - MEASUREMENT_SMOOTHING)
                            * smoothed_measurement
                        )

                    # Display smoothing.
                    if smoothed_display is None:

                        smoothed_display = corrected.copy()

                    else:

                        smoothed_display = (
                            DISPLAY_SMOOTHING * corrected
                            + (1 - DISPLAY_SMOOTHING)
                            * smoothed_display
                        )

                    x_measure, y_measure = smoothed_measurement

                    current_zone = zone_at(
                        x_measure,
                        y_measure,
                        width,
                        height,
                    )

                    if (
                        previous_time is not None
                        and previous_zone is not None
                    ):

                        elapsed = now - previous_time

                        if elapsed <= MAX_SAMPLE_GAP:
                            dwell[previous_zone] += elapsed

                    previous_time = now
                    previous_zone = current_zone

                    overlay.update_gaze(
                        *smoothed_display
                    )

                else:

                    smoothed_measurement = None
                    smoothed_display = None

                    previous_time = None
                    previous_zone = None

                    overlay.update_gaze(None)

                app.processEvents()

        finally:

            stop_event.set()

            if listener is not None and listener.is_alive():
                listener.stop()
                listener.join(timeout=1.0)

        return dwell

    finally:

        if listener is not None and listener.is_alive():
            listener.stop()
            listener.join(timeout=1.0)

        if calibration_window is not None:
            calibration_window.close()

        if validation_window is not None:
            validation_window.close()

        if recenter_button is not None:
            recenter_button.close()

        if overlay is not None:
            overlay.close()

        if camera_1 is not None:
            camera_1.release()

        if camera_2 is not None:
            camera_2.release()

        for estimator in estimators:
            estimator.close()

        app.processEvents()

        cv2.destroyAllWindows()


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Two-camera visual attention tracker"
    )

    parser.add_argument(
        "--camera1",
        type=int,
        default=0,
        help="First camera index",
    )

    parser.add_argument(
        "--camera2",
        type=int,
        default=1,
        help="Second camera index",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("attention_results.csv"),
        help="CSV output path",
    )

    args = parser.parse_args()

    if args.camera1 == args.camera2:
        parser.error(
            "Camera indices must be different."
        )

    try:

        dwell = track(
            args.camera1,
            args.camera2,
            args.output,
        )

        export_csv(
            args.output,
            dwell,
        )

        print("\nFinal attention results:")

        for zone, seconds in dwell.items():
            print(f"{zone}: {seconds:.3f} seconds")

        print(
            f"\nResults saved to: {args.output.resolve()}"
        )

    except KeyboardInterrupt:

        print("\nTracking cancelled.")

    except Exception as error:

        print(f"\nError: {error}")
        raise


if __name__ == "__main__":
    main()
```