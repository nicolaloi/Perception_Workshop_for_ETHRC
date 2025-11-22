import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import sys
import tqdm
sys.path.append('..')

from VisualOdometry import VisualOdometry
from VOPlotter import VOPlotter

# TODO: choose the section index (1 to 3)
LOAD_SECTION_IDX = 1
# TODO: choose the frame skipping interval (2 should be good)
SKIP_FRAME_INTERVAL = 2  # Process every N frames

USE_3D_SUBPLOT = False  # Set to False to show 3D points in BEV (2D top view) instead
SHOW_ONLY_INLIER_MATCHES = True  # Show only inlier feature matches in the visualization


# Load camera projection matrix (KITTI format)
projection_path = 'images/projection_matrix.csv'
cwd = os.getcwd()
block_dir = os.path.dirname(os.path.abspath(__file__))
projection_path = os.path.join(block_dir, projection_path)
with open(projection_path, 'r') as f:
    lines = f.readlines()
    # Parse the projection matrix values (skipping the comment line)
    proj_params = lines[1].strip().split(',')
    P_values = np.array([float(x) for x in proj_params])
# Extract intrinsic matrix K from the left 3x3 block of P
# For a monocular camera (reference camera), P = K @ [I | 0]
# So K is just the left 3x3 part of P
K = P_values.reshape(3, 4)[:3, :3]

print(f"\nExtracted Camera Intrinsic Matrix K (3x3):")
print(K)
print(f"\nCamera parameters:")
print(f"  Focal length: fx={K[0, 0]:.2f}, fy={K[1, 1]:.2f}")
print(f"  Principal point: cx={K[0, 2]:.2f}, cy={K[1, 2]:.2f}")

# Load images
images_dir = os.path.join(block_dir, f'images/{LOAD_SECTION_IDX}_section')
image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')))
print(f"\nFound {len(image_paths)} images in 'images/' directory.")
images = []
images_gray = []

for img_path in tqdm.tqdm(image_paths, desc="Loading images"):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_rgb)
    images_gray.append(img_gray)

print(f"\nLoaded {len(images)} images")
print(f"Image size: {images[0].shape[:2]}")


frames = images_gray[::SKIP_FRAME_INTERVAL]

vo = VisualOdometry(K)
vo_plotter = VOPlotter(vo, len(frames),
                       use_3d_subplot=USE_3D_SUBPLOT,
                       max_point_distance=250.0,
                       show_only_inlier_matches=SHOW_ONLY_INLIER_MATCHES,
                       save_video=True,
                       output_path=os.path.join(block_dir, 'trajectory_video.mp4'),
                       fps=50)

trajectory_points = []

tqdm_object = tqdm.tqdm(enumerate(frames), total=len(frames))
for img_id, img in tqdm_object:
	tqdm_object.set_description(f"Processing frame {img_id+1}/{len(frames)}")

	vo.update(img)
	x, y, z = vo.cur_t.flatten()
	trajectory_points.append((x, y, z))
	
	vo_plotter.plot(img_id, trajectory_points)

# Release video writer
vo_plotter.release()

plt.ioff()  # Disable interactive mode
plt.savefig(f'{block_dir}/trajectory_visualization.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to 'trajectory_visualization.png'")
print(f"Total 3D points triangulated: {len(vo.points_3d)}")
plt.show()


