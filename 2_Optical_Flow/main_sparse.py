import cv2
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import tqdm
from enum import Enum

import sys
import os
sys.path.append('..')
from viz_utils import VideoManager

cwd = os.getcwd()
block_dir = pathlib.Path(__file__).parent
videos_path = block_dir / 'videos'
print(f"Videos path: {videos_path}")

class VideoType(Enum):
    WALKING = 'walking.mp4'
    CARS = 'cars.mp4'
    SHEEPS = 'sheeps.mp4'
    DANCE = 'dance.mp4'

# TODO: choose the video to process
VIDEO = VideoType.WALKING

FRAME_INTERVAL_ADD_NEW_KEYPOINTS = 5  # Add new keypoints every 5 frames
DISPLACEMENT_PIXEL_THRESHOLD = 10.0  # Filter points with displacement higher than this threshold

# Create color mapping dictionary to keep colors consistent
color_map = {}
next_color_id = 0
max_colors = 10000
available_colors = np.random.randint(0, 255, (max_colors, 3))

# History tracking
max_viz_point_history = 30  # Keep last 20 frames of trajectory for each point
point_trajectories = {}  # Dictionary mapping color_id to list of points (trajectory)

video_path = videos_path / VIDEO.value
output_path = block_dir / f'output_sparse_{VIDEO.value}'

# TODO: implement the function to detect new keypoints to track
# have a look at cv2.goodFeaturesToTrack and play with its parameters!
def TODO_detect_new_keypoints_to_track(frame: np.ndarray) -> np.ndarray:
    # Compute new keypoint
    # Check https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html
    random_keypoints_v = np.random.randint(0, frame.shape[0], int(100)).astype(np.float32)
    random_keypoints_u = np.random.randint(0, frame.shape[1], int(random_keypoints_v.shape[0])).astype(np.float32)
    random_new_current_features = np.stack((random_keypoints_u, random_keypoints_v), axis=-1)

    return random_new_current_features

# TODO: implement the function to compute sparse optical flow
# have a look at cv2.calcOpticalFlowPyrLK and play with its parameters!
def TODO_compute_sparse_optical_flow(prev_image: np.ndarray, curr_image: np.ndarray, prev_points: np.ndarray) -> np.ndarray:
    # Compute the sparse optical flow using cv2.calcOpticalFlowPyrLK
    # Check https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    random_new_points = prev_points + np.random.randint(-5, 5, prev_points.shape).astype(np.float32)
    random_status = np.ones((prev_points.shape[0],1), dtype=np.uint8)

    return random_new_points, random_status



# Create VideoManager
with VideoManager(video_path, output_path, save_frames=False, show_preview=True) as video_mgr:
    # Take first frame
    ret, old_frame = video_mgr.read_frame()
    if not ret:
        raise ValueError("Could not read first frame")
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize empty tracking arrays
    p0 = np.array([]).reshape(0, 1, 2)
    keypoint_colors = []
    
    try:
        while True:
            ret, frame = video_mgr.read_frame()
            if not ret:
                print('Finished!')
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # calculate optical flow (skip if no points to track yet)
            if len(p0) > 0:
                p1, st = TODO_compute_sparse_optical_flow(old_gray, frame_gray, p0)
                
                # Select good points and their corresponding colors
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    # filter points with displacement higher than a threshold
                    displacement = np.linalg.norm(good_new - good_old, axis=1)
                    valid_displacement = displacement < DISPLACEMENT_PIXEL_THRESHOLD

                    good_new = good_new[valid_displacement]
                    good_old = good_old[valid_displacement]
                    good_colors = [keypoint_colors[i] for i, s in enumerate(st.flatten()) if s == 1]
                    good_colors = [color for i, color in enumerate(good_colors) if valid_displacement[i]]
                    
                    # Update trajectories for tracked points
                    for i, color_id in enumerate(good_colors):
                        new_point = tuple(good_new[i])
                        if color_id in point_trajectories:
                            point_trajectories[color_id].append(new_point)
                            # Keep only the last max_viz_point_history points
                            if len(point_trajectories[color_id]) > max_viz_point_history:
                                point_trajectories[color_id] = point_trajectories[color_id][-max_viz_point_history:]

                    # Remove trajectories for lost points
                    tracked_color_ids = set(good_colors)
                    lost_color_ids = set(point_trajectories.keys()) - tracked_color_ids
                    for color_id in lost_color_ids:
                        del point_trajectories[color_id]

                    # remove trajectories for points that barely moved during their history
                    for color_id in list(point_trajectories.keys()):
                        trajectory = point_trajectories[color_id]
                        if len(trajectory) >= max_viz_point_history / 2:
                            start_pt = np.array(trajectory[0])
                            end_pt = np.array(trajectory[-1])
                            dist_moved = np.linalg.norm(end_pt - start_pt)
                            if dist_moved < 1.0:
                                del point_trajectories[color_id]
            else:
                # No points to track yet
                good_new = np.array([]).reshape(0, 2)
                good_old = np.array([]).reshape(0, 2)
                good_colors = []
            
            # Add new keypoints every FRAME_INTERVAL_ADD_NEW_KEYPOINTS frames or if no keypoints exist
            if video_mgr.frame_idx % FRAME_INTERVAL_ADD_NEW_KEYPOINTS == 0 or len(good_new) == 0:
                # Detect new corners
                p0_new = TODO_detect_new_keypoints_to_track(frame_gray)
                
                if p0_new is not None:
                    # Filter out new keypoints that are too close to existing ones
                    p0_new_filtered = []
                    new_colors = []
                    
                    for new_pt in p0_new:
                        new_pt_coord = new_pt.ravel()
                        
                        # Check distance to existing keypoints only if there are any
                        if len(good_new) > 0:
                            # Calculate distances to all existing keypoints
                            distances = np.sqrt(np.sum((good_new.reshape(-1, 2) - new_pt_coord)**2, axis=1))
                            min_dist = np.min(distances)
                            
                            # Only add if far enough from all existing keypoints
                            if min_dist > 15:
                                p0_new_filtered.append(new_pt)
                                # Assign a new color
                                color_map[next_color_id] = available_colors[next_color_id % max_colors]
                                new_colors.append(next_color_id)
                                # Initialize trajectory for new point
                                point_trajectories[next_color_id] = [tuple(new_pt.ravel())]
                                next_color_id += 1
                        else:
                            # No existing keypoints, add all detected ones
                            p0_new_filtered.append(new_pt)
                            # Assign a new color
                            color_map[next_color_id] = available_colors[next_color_id % max_colors]
                            new_colors.append(next_color_id)
                            # Initialize trajectory for new point
                            point_trajectories[next_color_id] = [tuple(new_pt.ravel())]
                            next_color_id += 1
                    
                    # Add filtered new keypoints to the tracking list
                    if len(p0_new_filtered) > 0:
                        p0_new_filtered = np.array(p0_new_filtered)
                        if len(good_new) > 0:
                            good_new = np.vstack([good_new, p0_new_filtered.reshape(-1, 2)])
                            good_old = np.vstack([good_old, p0_new_filtered.reshape(-1, 2)])
                        else:
                            good_new = p0_new_filtered.reshape(-1, 2)
                            good_old = p0_new_filtered.reshape(-1, 2)
                        good_colors.extend(new_colors)
            
            # Create a fresh frame for drawing (no accumulated mask)
            draw_frame = frame.copy()
            
            # Draw the tracks using trajectory history
            for color_id in good_colors:
                if color_id in point_trajectories:
                    trajectory = point_trajectories[color_id]
                    point_color = color_map[color_id].tolist()
                    
                    # Draw lines connecting the trajectory points
                    for j in range(1, len(trajectory)):
                        pt1 = (int(trajectory[j-1][0]), int(trajectory[j-1][1]))
                        pt2 = (int(trajectory[j][0]), int(trajectory[j][1]))
                        cv2.line(draw_frame, pt1, pt2, point_color, 2)
                    
                    # Draw current point as a circle
                    if len(trajectory) > 0:
                        current_pt = (int(trajectory[-1][0]), int(trajectory[-1][1]))
                        cv2.circle(draw_frame, current_pt, 5, point_color, -1)
            
            # Write frame to video (with frame counter)
            video_mgr.write_frame(draw_frame, add_frame_counter=True)
            
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            keypoint_colors = good_colors
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")

print("Processing complete!")