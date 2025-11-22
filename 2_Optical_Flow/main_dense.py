import cv2 # opencv is a computer vision library
import numpy as np # numpy is a library for numerical computing
from matplotlib import pyplot as plt # matplotlib is a plotting library
import pathlib
import tqdm
from enum import Enum

import sys
import os
sys.path.append('..')
from viz_utils import visualize_dense_optical_flow, VisualizationType, VideoManager

cwd = os.getcwd()
block_dir = pathlib.Path(__file__).parent
videos_path = block_dir / 'videos'
print(f"Videos path: {videos_path}")

class VideoType(Enum):
    WALKING = 'walking.mp4'
    CARS = 'cars.mp4'
    SHEEPS = 'sheeps.mp4'
    DANCE = 'dance.mp4'

# TODO Choose what video to process, and the visualization type
VIDEO = VideoType.WALKING
VISUALIZATION = VisualizationType.HSV_COLOR

# TODO: implement the function to compute dense optical flow
# have a look at cv2.calcOpticalFlowFarneback and play with its parameters!
def TODO_calculate_dense_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray, prev_flow: np.ndarray) -> np.ndarray:
    # use the OpenCV Farneback method to compute dense optical flow: cv2.calcOpticalFlowFarneback
    # check https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
    random_flow = np.random.randn(prev_frame.shape[0]//10, prev_frame.shape[1]//10, 2).astype(np.float32)
    random_flow = cv2.resize(random_flow, (prev_frame.shape[1], prev_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return random_flow


video_path = videos_path / VIDEO.value
output_path = block_dir / f'output_dense_{VISUALIZATION.value}_{VIDEO.value}'
flow = None

# Create VideoManager
with VideoManager(video_path, output_path, save_frames=False, show_preview=True) as video_mgr:
    # Read first frame
    ret, frame1 = video_mgr.read_frame()
    if not ret:
        raise ValueError("Could not read first frame")
    
    prev_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Process frames
    try:
        while True:
            ret, frame = video_mgr.read_frame()
            if not ret:
                print('Finished!')
                break
            
            next_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = TODO_calculate_dense_optical_flow(prev_image, next_image, flow)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            out_frame = visualize_dense_optical_flow(frame, flow, mag, ang, VISUALIZATION)
            
            # Write frame to video (with frame counter)
            video_mgr.write_frame(out_frame, add_frame_counter=True)

            prev_image = next_image

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")

print("Processing complete!")