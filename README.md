# Attention Tracking

Local webcam-based visual attention prototype built with Python and [EyeTrax](https://github.com/ck-zhang/EyeTrax).

The application calibrates gaze estimation, overlays a semi-transparent visualization on the desktop, divides the primary display into four zones, measures how long the user looks at each zone, and exports the results as CSV.

The overlay stays above other application windows, allowing users to observe their gaze distribution while interacting with other content. Camera frames are processed locally and are not saved.

## How it works

1. EyeTrax performs a 9-point calibration using webcam face and eye landmarks.
2. The calibrated model predicts screen coordinates for each valid camera frame.
3. Exponential smoothing reduces movement in the predicted gaze point.
4. Each point is assigned to one of four screen zones.
5. Valid consecutive samples contribute to the corresponding zone's dwell time.
6. A transparent, always-on-top overlay visualizes the four zones and current gaze position.
7. Pressing Escape ends the session and writes the CSV summary.

## Visualization

The application uses a transparent desktop overlay instead of an opaque fullscreen window.

- Four colored regions represent the screen zones.
- A green circle indicates the estimated gaze position.
- The overlay remains above other application windows.
- Mouse events can pass through the overlay, allowing interaction with underlying applications.
- Zone opacity can be adjusted independently of the underlying desktop content.

The default opacity is 20%, providing a subtle visualization without completely obscuring the content underneath.

### Zone layout

| Zone | Screen area |
| --- | --- |
| `zone_1` | Top-left |
| `zone_2` | Top-right |
| `zone_3` | Bottom-left |
| `zone_4` | Bottom-right |

## Requirements

- Python 3.10 or newer
- Webcam
- One primary monitor
- Windows, macOS, or Linux with a graphical desktop
- PySide6 for transparent overlay rendering

## Setup

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt`:

```text
eyetrax
opencv-python
numpy
screeninfo
PySide6
pynput
```

## Run

```bash
python attention_tracker.py
```

Look directly at each calibration point. After calibration, the transparent overlay appears over the desktop.

Look around the four zones while using other applications or viewing images. Press **Escape** to finish and create `attention_results.csv`.

### Optional arguments

Select a different camera:

```bash
python attention_tracker.py --camera 1
```

Specify a different CSV output location:

```bash
python attention_tracker.py --output results/session_1.csv
```

## Overlay opacity

The overlay opacity controls how strongly the colored zones appear over underlying applications.

| Opacity | Appearance |
| --- | --- |
| `0.10` | Very subtle |
| `0.20` | Light (default) |
| `0.35` | Clearly visible |
| `0.50` | Strong |

Opacity is configured in the overlay initialization.

## CSV output

The application exports a CSV file containing the dwell time and percentage for each zone.

```csv
zone,dwell_seconds,percentage,most_looked_at
zone_1,5.283,25.93,False
zone_2,7.718,37.88,True
zone_3,2.706,13.28,False
zone_4,4.667,22.91,False
```

`most_looked_at=True` identifies the zone with the longest recorded dwell time.

## Privacy

- Webcam frames are processed locally.
- Camera frames are not saved.
- Gaze measurements are exported only as aggregated zone dwell times.
- No cloud processing or external data transmission is required.

## Platform compatibility

Transparent, always-on-top overlays depend on the operating system and desktop environment.

- **Windows:** Supports transparent, always-on-top windows and click-through behavior.
- **macOS:** May require additional window configuration and permissions.
- **Linux:** Behavior depends on the window manager and display server. X11 generally offers better compatibility than Wayland.

Exclusive fullscreen applications and games may prevent the overlay from appearing above their content.