# Attention Tracking

Local webcam-based visual attention prototype built with Python and [EyeTrax](https://github.com/ck-zhang/EyeTrax).

The application calibrates gaze estimation, divides the primary display into four zones, measures how long the user looks at each zone, and exports the result as CSV. Camera frames are processed locally and are not saved.

## How it works

1. EyeTrax performs a 9-point calibration using webcam face and eye landmarks.
2. The calibrated model predicts screen coordinates for each valid camera frame.
3. Exponential smoothing reduces movement in the predicted point.
4. Each point is assigned to one of four screen zones.
5. Valid consecutive samples contribute to the zone's dwell time.
6. Pressing Escape ends the session and writes the CSV summary.

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

## Setup

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```bash
python attention_tracker.py
```

Look directly at each calibration point. After calibration, look around the four zones. Press *Escape* to finish and create `attention_results.csv`.

Optional arguments:

```bash
python attention_tracker.py --camera 1
python attention_tracker.py --output results/session_1.csv
```

## CSV output

```csv
zone,dwell_seconds,percentage,most_looked_at
zone_1,5.283,25.93,False
zone_2,7.718,37.88,True
zone_3,2.706,13.28,False
zone_4,4.667,22.91,False
```

`most_looked_at=True` identifies the zone with the longest recorded dwell time.

## Visual work


https://drive.google.com/file/d/19w-m84G8inJzrRGtPvx7Oc7F3bUC8M_u/view?usp=sharing





