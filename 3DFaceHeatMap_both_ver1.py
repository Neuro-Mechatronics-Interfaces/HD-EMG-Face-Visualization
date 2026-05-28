import numpy as np
import pyvista as pv
import scipy.io
import time
from scipy.spatial import cKDTree
import tkinter as tk
from tkinter import filedialog
import os
os.environ['PYVISTA_USE_PANEL'] = 'false'
pv.global_theme.multi_rendering_splitting_position = 0.5

# Force software rendering fallback
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
pv.global_theme.render_lines_as_tubes = False
pv.global_theme.smooth_shading = False

# ── File selection ────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

print("Please select the LEFT side EMG data file...")
left_path = filedialog.askopenfilename(
    title="Select LEFT side EMG .mat file",
    filetypes=[("MATLAB Files", "*.mat *.pomat"), ("All Files", "*.*")]
)
if not left_path:
    print("No file selected. Exiting.")
    exit()

print("Please select the RIGHT side EMG data file...")
right_path = filedialog.askopenfilename(
    title="Select RIGHT side EMG .mat file",
    filetypes=[("MATLAB Files", "*.mat *.pomat"), ("All Files", "*.*")]
)
if not right_path:
    print("No file selected. Exiting.")
    exit()

print(f"Left:  {left_path}")
print(f"Right: {right_path}")

# ── Active segments ───────────────────────────────────────────────────
segments = [
    (15, 35),
    (45, 65),
]

# ── Function to load and precompute frames ────────────────────────────
def load_frames(filepath, segments, target_fps=30, speed_factor=4):
    mat = scipy.io.loadmat(filepath)
    SIG = mat['SIG']
    discard = mat['discardChannelsVec']
    fsamp = int(mat['fsamp'][0][0])

    step = int(fsamp / target_fps) * speed_factor
    window = int(0.03 * fsamp)
    segment_samples = [(int(s * fsamp), int(e * fsamp)) for s, e in segments]

    all_frames = []
    for seg_start, seg_end in segment_samples:
        for start in range(seg_start, seg_end - window, step):
            end = start + window
            frame_vals = []
            for row in range(8):
                for col in range(4):
                    if discard[row, col] == 1:
                        frame_vals.append(np.nan)
                    else:
                        chunk = SIG[row, col][0][start:end]
                        frame_vals.append(np.sqrt(np.mean(chunk**2)))
            frame_vals = np.array(frame_vals)
            frame_vals = np.where(np.isnan(frame_vals), np.nanmean(frame_vals), frame_vals)
            all_frames.append(frame_vals)

    return np.array(all_frames)

# ── Load both sides ───────────────────────────────────────────────────
print("Loading left side...")
left_frames = load_frames(left_path, segments)

print("Loading right side...")
right_frames = load_frames(right_path, segments)

# Match frame counts (use minimum)
n_frames = min(len(left_frames), len(right_frames))
left_frames  = left_frames[:n_frames]
right_frames = right_frames[:n_frames]

# Global color range across both sides
all_data = np.concatenate([left_frames, right_frames])
vmin = np.percentile(all_data, 5)
vmax = np.percentile(all_data, 95)
target_fps = 30

left_frames  = np.clip(left_frames,  vmin, vmax)
right_frames = np.clip(right_frames, vmin, vmax)

print(f"Total frames: {n_frames} ({n_frames/target_fps:.1f}s)")
print(f"Activation range: {vmin:.1f} to {vmax:.1f}")

# ── Load meshes ───────────────────────────────────────────────────────
folder = r'C:\Users\yelim\Documents\Lab\Face Model\FabFaceMuscles'
muscle_mesh = pv.read(folder + r'\SubTool-0-2979770.obj')
skin_mesh   = pv.read(folder + r'\SubTool-1-5430089.obj')

muscle_points = np.array(muscle_mesh.points)
muscle_mesh.compute_normals(inplace=True)
muscle_normals = np.array(muscle_mesh.point_normals)
mesh_center = muscle_points.mean(axis=0)

tree = cKDTree(muscle_points)

# ── Function to build heatmap plane ──────────────────────────────────
def build_plane(top_left, col_dir, row_dir, electrode_normals, offset=0.15, curve=0.5):
    scale_factor = 1.96 / 140.0
    ied_units = 8.75 * scale_factor
    plane_rows, plane_cols = 8, 4

    plane_points = []
    for row in range(plane_rows):
        for col in range(plane_cols):
            point = top_left + (col * ied_units * col_dir) + (row * ied_units * row_dir)
            plane_points.append(point)
    plane_points = np.array(plane_points)

    # Average normal for offset direction
    normal_avg = electrode_normals.mean(axis=0)
    normal_avg = normal_avg / np.linalg.norm(normal_avg)

    # Curved offset
    plane_center = plane_points.mean(axis=0)
    plane_xyz_offset = np.zeros_like(plane_points)
    for i, pt in enumerate(plane_points):
        dist_col = np.dot(pt - plane_center, col_dir)
        dist_row = np.dot(pt - plane_center, row_dir)
        c = curve * (dist_col**2 + dist_row**2)
        plane_xyz_offset[i] = pt + (offset - c) * normal_avg

    # Build faces
    faces = []
    for row in range(plane_rows - 1):
        for col in range(plane_cols - 1):
            i00 = row * plane_cols + col
            i10 = (row + 1) * plane_cols + col
            i01 = row * plane_cols + (col + 1)
            i11 = (row + 1) * plane_cols + (col + 1)
            faces.extend([4, i00, i01, i11, i10])

    return plane_points, plane_xyz_offset, np.array(faces)

# ── Left side grid ────────────────────────────────────────────────────
left_top_left = np.array([0.545, -0.355, -0.759])
left_col_dir  = np.array([ 0.0,   0.327, -0.945])
left_row_dir  = np.array([-0.04, -0.972, -0.233])

left_grid_points = []
scale_factor = 1.96 / 140.0
ied_units = 8.75 * scale_factor
for row in range(8):
    for col in range(4):
        pt = left_top_left + (col * ied_units * left_col_dir) + (row * ied_units * left_row_dir)
        left_grid_points.append(pt)
left_grid_points = np.array(left_grid_points)

_, left_nearest = tree.query(left_grid_points)
left_electrode_xyz = muscle_points[left_nearest]
left_electrode_normals = muscle_normals[left_nearest].copy()
for i, (pt, n) in enumerate(zip(left_electrode_xyz, left_electrode_normals)):
    if np.dot(n, pt - mesh_center) < 0:
        left_electrode_normals[i] = -n

_, left_plane_xyz, left_faces = build_plane(
    left_top_left, left_col_dir, left_row_dir, left_electrode_normals
)

# ── Right side grid ───────────────────────────────────────────────────
# Mirror the left side: flip X axis
right_top_left = left_top_left * np.array([-1, 1, 1])
right_col_dir  = left_col_dir  * np.array([-1, 1, 1])
right_row_dir  = left_row_dir  * np.array([-1, 1, 1])

right_grid_points = []
for row in range(8):
    for col in range(4):
        pt = right_top_left + (col * ied_units * right_col_dir) + (row * ied_units * right_row_dir)
        right_grid_points.append(pt)
right_grid_points = np.array(right_grid_points)

_, right_nearest = tree.query(right_grid_points)
right_electrode_xyz = muscle_points[right_nearest]
right_electrode_normals = muscle_normals[right_nearest].copy()
for i, (pt, n) in enumerate(zip(right_electrode_xyz, right_electrode_normals)):
    if np.dot(n, pt - mesh_center) < 0:
        right_electrode_normals[i] = -n

_, right_plane_xyz, right_faces = build_plane(
    right_top_left, right_col_dir, right_row_dir, right_electrode_normals
)

# ── Build heatmap planes ──────────────────────────────────────────────
left_plane  = pv.PolyData(left_plane_xyz,  left_faces)
right_plane = pv.PolyData(right_plane_xyz, right_faces)

left_plane['EMG Activation']  = left_frames[0]
right_plane['EMG Activation'] = right_frames[0]

# ── Set up plotter ────────────────────────────────────────────────────
p = pv.Plotter(lighting='three lights')
p.add_mesh(skin_mesh,   color='white',   opacity=1.0)
p.add_mesh(muscle_mesh, color='#FFB6C1', opacity=1.0)

p.add_mesh(
    left_plane,
    scalars='EMG Activation',
    cmap='hot',
    opacity=1.0,
    clim=[vmin, vmax],
    scalar_bar_args={'title': 'RMS Activation', 'vertical': True}
)
p.add_mesh(
    right_plane,
    scalars='EMG Activation',
    cmap='hot',
    opacity=1.0,
    clim=[vmin, vmax],
    show_scalar_bar=False
)

p.set_background('white')
p.add_text('EMG Heatmap — Bilateral', font_size=12, color='black')

# ── Animate ───────────────────────────────────────────────────────────
p.show(auto_close=False, interactive_update=True)

frame_state = 0
while True:
    left_plane['EMG Activation']  = left_frames[frame_state]
    right_plane['EMG Activation'] = right_frames[frame_state]
    p.update()
    time.sleep(1.0 / target_fps)
    frame_state = (frame_state + 1) % n_frames