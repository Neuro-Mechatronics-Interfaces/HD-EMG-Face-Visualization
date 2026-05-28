# HD-EMG Facial Muscle Heatmap Visualization

A Python-based visualization pipeline for displaying high-density surface EMG (HD-EMG) 
activation as animated heatmaps on a 3D facial muscle model. Developed at the 
Neuro-Mechatronics Interfaces Lab, Carnegie Mellon University.

## Overview

This tool takes DECOMPOSED HD-EMG data (MAT file) from a TMSi textile electrode grid placed over 
the zygomaticus/buccinator/masseter region and displays real-time RMS activation as an 
animated color heatmap overlaid on a 3D facial muscle model. Designed for clinical 
interpretation by physicians.

## Features

- Animated heatmap playing on a 3D facial muscle model
- Bilateral support (left and right side simultaneously)
- File dialog for easy data selection
- Configurable active segment extraction (removes rest periods)
- Hot colormap with consistent color scaling across both sides

## Requirements

### Python packages

```
pip install pyvista scipy numpy
```

### Required files

- **3D muscle model**: "Head Study: Muscles" from Fab.com
- Please reach out to obtain the OBJ files.
  Place OBJ files in a folder and update the `folder` path in the script:
  - `SubTool-0-2979770.obj` — muscles
  - `SubTool-1-5430089.obj` — skull
- **EMG data**: DECOMPOSED HD-EMG `.mat` or `.pomat` files from TMSi SAGA  
  with the following fields: `SIG`, `discardChannelsVec`, `fsamp`

## Hardware

- **Amplifier**: TMSi SAGA
- **Electrode grid**: TMSi textile HD-EMG grid (8 rows x 4 columns, 8.75mm IED)
- **Placement**: Zygomaticus major / buccinator / masseter region

## Usage

1. Clone this repository:

```
git clone https://github.com/Neuro-Mechatronics-Interfaces/facial-emg-heatmap.git
```

### 2. 3D Facial Muscle Model (required, not included)

This project uses the **"Head Study: Muscles"** model from Fab.com.

1. Go to: https://www.fab.com/listings/72bba72c-07fe-45a7-b4b3-8cc5496f6404
2. Purchase and download the model (~$12.99)
3. Unzip and locate the OBJ files:
   - `SubTool-0-2979770.obj` — muscles (required)
   - `SubTool-1-5430089.obj` — skull (required)
   - `SubTool-4-2377246.obj` — eyeballs (not used)
4. Place them in a folder on your machine
5. Update this line in the script to point to that folder:

```python

3. Run the script:

```
python emg_heatmap.py
```

4. When prompted, select:
   - Your **left side** EMG `.mat` file
   - Your **right side** EMG `.mat` file

5. The 3D viewer will open with the animated heatmap playing automatically.
   You can rotate, zoom, and pan the model freely while the heatmap plays.

## Configuration

At the top of the script, adjust these parameters to match your protocol:

```python
# Active segments to visualize (in seconds)
segments = [
    (15, 35),   # rep 1: ramp up + gesture + ramp down
    (45, 65),   # rep 2: ramp up + gesture + ramp down
]

# Electrode grid anchor points
left_top_left = np.array([0.545, -0.355, -0.759])
left_col_dir  = np.array([ 0.0,   0.327, -0.945])
left_row_dir  = np.array([-0.04, -0.972, -0.233])
```

## Data Format

Input `.mat` files should be decomposed HD-EMG files with the following structure:

| Field | Description |
|---|---|
| `SIG` | (8x8) object array — each cell contains a (1, N) time series |
| `discardChannelsVec` | (8x8) binary matrix — 1 = bad channel |
| `fsamp` | Sampling frequency (Hz) |

## Electrode Grid Placement

The grid is placed over the left zygomaticus major / buccinator region, running 
diagonally from the cheekbone toward the jaw. The right side is automatically 
mirrored across the X axis.

To re-calibrate electrode placement for a different subject or anatomy, use the 
interactive picker:

```python
p.enable_point_picking(callback=callback, show_message=True)
```

Click three points: top-left corner, top-right corner, and bottom-left corner of 
the grid. The script will compute top_left, col_dir, and row_dir automatically.

## Known Issues

- PyVista 0.47.2: add_timer_event is non-functional — animation uses a while loop
  with interactive_update=True instead
- PyVista 0.47.2: points must be rendered as pv.Sphere() geometry rather than add_points()
- Shader creation errors may occur depending on GPU driver — add this workaround:

```python
import os
os.environ['PYVISTA_USE_PANEL'] = 'false'
pv.global_theme.render_lines_as_tubes = False
pv.global_theme.smooth_shading = False
```

## Authors

Yelim Ki — Carnegie Mellon University  
Neuro-Mechatronics Interfaces Lab
