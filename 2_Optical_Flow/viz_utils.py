import cv2
from matplotlib.pyplot import hsv
import numpy as np
from enum import Enum

class VisualizationType(Enum):
    HSV_COLOR = 'hsv'
    ARROWS = 'arrows'
    BOTH = 'both'


import cv2
import numpy as np
import pathlib
import tqdm
from typing import Optional, List


class VideoManager:
    """
    A class to manage video processing, including reading frames, writing output,
    and displaying results.
    """
    
    def __init__(self, input_path: pathlib.Path, output_path: pathlib.Path, 
                 save_frames: bool = False, show_preview: bool = True):
        """
        Initialize the VideoManager.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            save_frames: Whether to store frames in memory
            show_preview: Whether to show live preview window
        """
        self.input_path = input_path
        self.output_path = output_path
        self.save_frames = save_frames
        self.show_preview = show_preview
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(input_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_video = cv2.VideoWriter(str(output_path), fourcc, self.fps, 
                                         (self.width, self.height))
        
        # Storage for frames if needed
        self.frames: List[np.ndarray] = [] if save_frames else None
        
        # Frame counter
        self.frame_idx = 0
        
        print(f"Video properties:")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
    
    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video.
        
        Returns:
            Tuple of (success, frame)
        """
        ret, frame = self.cap.read()
        if ret:
            self.frame_idx += 1
        return ret, frame
    
    def write_frame(self, frame: np.ndarray, add_frame_counter: bool = True,
                   text_color: tuple = (0, 255, 0)) -> None:
        """
        Write a frame to the output video and optionally display it.
        
        Args:
            frame: Frame to write
            add_frame_counter: Whether to add frame counter text
            text_color: Color of the frame counter text (BGR)
        """
        # Make a copy if we need to add text (to avoid modifying original)
        output_frame = frame.copy() if add_frame_counter or self.show_preview else frame
        
        # Add frame counter
        if add_frame_counter:
            cv2.putText(output_frame, 
                       f'Frame: {self.frame_idx}/{self.total_frames}', 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, 
                       text_color, 
                       2)
        
        # Write to video file
        self.out_video.write(output_frame)
        
        # Store frame if requested
        if self.save_frames:
            self.frames.append(output_frame)
        
        # Show preview
        if self.show_preview:
            cv2.imshow('Video Preview', output_frame)
            key = cv2.waitKey(1) & 0xff
            if key == 27:  # ESC key
                raise KeyboardInterrupt("User interrupted processing")
    
    def get_progress(self) -> float:
        """
        Get the current processing progress as a percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        return (self.frame_idx / self.total_frames) * 100 if self.total_frames > 0 else 0
    
    def release(self) -> None:
        """
        Release video resources and close windows.
        """
        self.cap.release()
        self.out_video.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        
        print(f"\nVideo saved to: {self.output_path}")
        print(f"Total frames processed: {self.frame_idx}")
    
    def save_frames_to_disk(self, output_dir: pathlib.Path, 
                           prefix: str = 'frame') -> None:
        """
        Save stored frames to individual image files.
        
        Args:
            output_dir: Directory to save frames
            prefix: Filename prefix for frames
        """
        if not self.save_frames or self.frames is None:
            print("No frames stored in memory. Set save_frames=True when initializing.")
            return
        
        output_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(tqdm.tqdm(self.frames, desc="Saving frames")):
            frame_path = output_dir / f'{prefix}_{i:06d}.png'
            cv2.imwrite(str(frame_path), frame)
        print(f"Individual frames saved to: {output_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False  # Don't suppress exceptions


def visualize_dense_optical_flow(frame, flow, mag, ang, viz_type: VisualizationType):

    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    # clip magnitude to filter out weird normalization effects for visualization
    MIN_PERCENTILE = 1.0
    MAX_PERCENTILE = 99.5
    mag_clipped = np.clip(mag, np.percentile(mag, MIN_PERCENTILE), np.percentile(mag, MAX_PERCENTILE))
    
    # Prepare output frame based on visualization type
    if viz_type == VisualizationType.HSV_COLOR or viz_type == VisualizationType.BOTH:
        # HSV color visualization
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag_clipped, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out_frame = cv2.addWeighted(frame, 0.4, bgr, 0.6, 0)
    else:
        out_frame = frame.copy()
    
    if viz_type == VisualizationType.ARROWS or viz_type == VisualizationType.BOTH:
        # Draw flow arrows (with color based on angle)
        out_frame = draw_flow_arrows(out_frame, flow, step=16, scale=5.0, thickness=2, color_by_angle=True)
    return out_frame


def draw_flow_arrows(img, flow, step=16, scale=1.5, color=(0, 255, 0), thickness=1, color_by_angle=False):
    """
    Draw optical flow as arrows on the image.
    
    Args:
        img: Input image
        flow: Optical flow (H, W, 2)
        step: Spacing between arrows in pixels
        scale: Arrow length scaling factor
        color: Arrow color (BGR) - used only if color_by_angle is False
        thickness: Arrow line thickness
        color_by_angle: If True, color arrows based on flow direction
    
    Returns:
        Image with arrows drawn
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    
    # Create lines for arrows
    lines = np.vstack([x, y, x + fx*scale, y + fy*scale]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # Draw arrows
    vis = img.copy()
    for i, ((x1, y1), (x2, y2)) in enumerate(lines):
        # Calculate arrow magnitude
        mag = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if mag > 1.0:  # Only draw if there's significant motion
            if color_by_angle:
                # Calculate angle for this flow vector
                angle = np.arctan2(fy[i], fx[i])
                # Convert angle to hue (0-180 for OpenCV)
                hue = int((angle * 180 / np.pi) % 360 / 2)
                # Create HSV color with full saturation and value
                hsv_color = np.uint8([[[hue, 255, 255]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                arrow_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
            else:
                arrow_color = color
            
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), arrow_color, thickness, tipLength=0.3)
    
    return vis