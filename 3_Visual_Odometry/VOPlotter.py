import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import sys
sys.path.append('..')

from VisualOdometry import VisualOdometry

# Increase DPI for better resolution based on your screen size
DPI = 60

class VOPlotter:
	def __init__(self, vo: VisualOdometry, n_images, use_3d_subplot=True, max_point_distance=50.0,
			  show_only_inlier_matches=False, save_video=False, output_path=None, fps=10):
		"""
		Initialize the Visual Odometry Plotter.
		
		Args:
			vo: VisualOdometry instance
			n_images: Total number of images to process
			use_3d_subplot: If True, creates a 3D subplot for 3D visualization.
			                If False, creates a zoomed BEV (2D top view) in the third subplot.
			max_point_distance: Maximum distance from camera to display 3D points (meters)
			save_video: Whether to save frames to video
			output_path: Path to output video file
			fps: Frames per second for output video
		"""
		self.vo = vo
		self.use_3d_subplot = use_3d_subplot
		self.n_images = n_images
		self.trajectory_points = []
		self.max_point_distance = max_point_distance
		self.show_only_inliers = show_only_inlier_matches
		self.save_video = save_video
		
		# Video writer setup
		self.video_writer = None
		if save_video and output_path is not None:
			self.output_path = output_path
			self.fps = fps
			# Video writer will be initialized after first frame is rendered

		if use_3d_subplot:
			# 3-plot layout: frame, BEV, and 3D view
			self.fig = plt.figure(figsize=(16, 10), dpi=DPI)
			self.ax1 = plt.subplot(2, 2, (1, 2))
			self.ax2 = plt.subplot(2, 2, 3)
			self.ax3 = plt.subplot(2, 2, 4, projection='3d')
		else:
			# 3-plot layout: frame, BEV, and zoomed BEV
			self.fig = plt.figure(figsize=(16, 10), dpi=DPI)
			self.ax1 = plt.subplot(2, 2, (1, 2))
			self.ax2 = plt.subplot(2, 2, 3)
			self.ax3 = plt.subplot(2, 2, 4)  # No 3D projection for zoom view
		
		# Setup static elements
		self.ax1.axis('off')
		self.ax2.set_xlabel('X (m)', fontsize=12)
		self.ax2.set_ylabel('Z (m)', fontsize=12)
		self.ax2.grid(True, alpha=0.3)
		
		if not use_3d_subplot:
			self.ax3.set_xlabel('X (m)', fontsize=12)
			self.ax3.set_ylabel('Z (m)', fontsize=12)
			self.ax3.grid(True, alpha=0.3)
		else:
			self.ax3.set_xlabel('X (m)', fontsize=10)
			self.ax3.set_ylabel('Z (m)', fontsize=10)
			self.ax3.set_zlabel('Y (m)', fontsize=10)
		
		if save_video:
			plt.ioff()  # Disable interactive mode when saving video
		else:
			plt.ion()   # Enable interactive mode for display
		plt.show(block=False)
		
		# Draw the figure once to initialize the background
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		
		# Store background for blitting
		self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

	def draw_tracks(self, frame, prev_features, curr_features, inliers=None):
		"""
		Draw feature tracks on the frame.
		
		Args:
			frame: Grayscale frame
			prev_features: Previous frame keypoints (N, 2)
			curr_features: Current frame keypoints (N, 2)
			inliers: Inlier mask (N,) - 1 for inliers, 0 for outliers
			show_only_inliers: If True, only draw inlier tracks
			
		Returns:
			Frame with tracks drawn (BGR image)
		"""
		out = frame.copy()
		out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
		
		if curr_features is None or prev_features is None:
			return out
			
		for i, (new, old) in enumerate(zip(curr_features, prev_features)):
			if self.show_only_inliers and inliers is not None and inliers[i] == 0:
				continue
			a, b = new.ravel()
			c, d = old.ravel()
			# Draw previous position as blue circle
			out = cv2.circle(out, (int(c), int(d)), 4, (0, 0, 255), -1)
			# Draw arrow from old to new position
			# Green for inliers, red for outliers
			color = (255, 0, 0) if inliers is not None and inliers[i] == 0 else (0, 255, 0)
			out = cv2.arrowedLine(out, (int(c), int(d)), (int(a), int(b)), color, 2, tipLength=0.1)
		
		return out

	def plot(self, img_id, trajectory_points):

		img = self.vo.new_frame
		points_3d = self.vo.get_recent_3d_points(max_points=200000, skip_interval=2)
		prev_features = self.vo.last_features
		curr_features = self.vo.new_features
		inliers = self.vo.inliers
		
		# Clear previous plots (we need to clear for simplicity)
		self.ax1.clear()
		self.ax2.clear()
		if self.ax3 is not None:
			self.ax3.clear()
		
		# Re-setup static elements after clearing
		self.ax1.axis('off')
		self.ax2.set_xlabel('X (m)', fontsize=12)
		self.ax2.set_ylabel('Z (m)', fontsize=12)
		self.ax2.grid(True, alpha=0.3)

		if points_3d is not None and len(points_3d) > 0 and len(trajectory_points) > 0:
			curr_pose = np.asarray(trajectory_points[-1])
			points_3d = points_3d[
				np.linalg.norm(points_3d - curr_pose, axis=1) < self.max_point_distance ]

		# Upper/First plot: Frame with tracks
		if img is not None:
			# Draw tracks on the frame
			frame_with_tracks = self.draw_tracks(img, prev_features, curr_features, inliers)
			frame_rgb = cv2.cvtColor(frame_with_tracks, cv2.COLOR_BGR2RGB)
			self.ax1.imshow(frame_rgb)
			self.ax1.set_title(f"Road Facing Camera - Frame {img_id+1}/{self.n_images}, "
					  f"{len(curr_features) if curr_features is not None else 0} current features, {np.sum(inliers)} inliers", fontsize=14, fontweight='bold')
			self.ax1.axis('off')

		# Draw feature tracks on the frame
		if img is not None and prev_features is not None and curr_features is not None:
			tracks_img = self.draw_tracks(img, prev_features, curr_features, inliers)
			self.ax1.imshow(tracks_img)
		
		# Second plot: Trajectory (Top View / BEV)
		if len(trajectory_points) > 1:
			traj_array = np.array(trajectory_points)
			x = traj_array[:, 0]
			z = traj_array[:, 2]
			self.ax2.plot(x, z, 'b-', linewidth=2, label='Estimated Path')
			self.ax2.scatter(traj_array[-1, 0], traj_array[-1, 2], c='red', s=100, marker='o', 
						edgecolors='yellow', linewidth=2, zorder=10, label='Current Position')
			self.ax2.scatter(0, 0, c='green', s=150, marker='s', label='Start', zorder=5)
			
			# Plot 3D points in BEV
			if points_3d is not None and len(points_3d) > 0:
				self.ax2.scatter(points_3d[:, 0], points_3d[:, 2], 
							   c='cyan', s=1, alpha=0.3, label='3D Points', zorder=1)

		x, y, z = trajectory_points[-1]
		num_3d_points = len(points_3d) if points_3d is not None else 0
		
		if self.use_3d_subplot:
			title = f'Camera Trajectory - Top View'
		else:
			title = f'Camera Trajectory & 3D Points - BEV ({num_3d_points} points)'
		
		self.ax2.set_xlabel('X (unscaled)', fontsize=12)
		self.ax2.set_ylabel('Z (unscaled)', fontsize=12)
		self.ax2.set_title(title, fontsize=14, fontweight='bold')
		self.ax2.grid(True, alpha=0.3)
		self.ax2.legend(loc='upper right')
		self.ax2.axis('equal')
		
		# Third plot: 3D visualization OR zoomed BEV
		if self.use_3d_subplot and self.ax3 is not None:
			# 3D visualization mode
			if len(trajectory_points) > 1:
				traj_array = np.array(trajectory_points)
				self.ax3.plot(traj_array[:, 0], traj_array[:, 2], traj_array[:, 1], 
							 'b-', linewidth=2, label='Camera Trajectory')
				self.ax3.scatter(traj_array[-1, 0], traj_array[-1, 2], traj_array[-1, 1], 
							   c='red', s=100, marker='o', edgecolors='yellow', linewidth=2, 
							   zorder=10, label='Current Position')
				self.ax3.scatter(0, 0, 0, c='green', s=150, marker='s', label='Start', zorder=5)
			
			# Plot 3D points
			if points_3d is not None and len(points_3d) > 0:
				self.ax3.scatter(points_3d[:, 0], points_3d[:, 2], points_3d[:, 1], 
							   c='cyan', s=1, alpha=0.3, label='3D Points')
			
			self.ax3.set_xlabel('X (unscaled)', fontsize=10)
			self.ax3.set_ylabel('Z (unscaled)', fontsize=10)
			self.ax3.set_zlabel('Y (unscaled)', fontsize=10)
			self.ax3.set_title(f'3D Reconstruction ({num_3d_points} points)', fontsize=14, fontweight='bold')
			self.ax3.legend(loc='upper right')
			
			# Set equal aspect ratio for better visualization
			if len(trajectory_points) > 1:
				traj_array = np.array(trajectory_points)
				max_range = np.array([traj_array[:, 0].max()-traj_array[:, 0].min(),
									 traj_array[:, 2].max()-traj_array[:, 2].min(),
									 traj_array[:, 1].max()-traj_array[:, 1].min()]).max() / 2.0
				mid_x = (traj_array[:, 0].max()+traj_array[:, 0].min()) * 0.5
				mid_z = (traj_array[:, 2].max()+traj_array[:, 2].min()) * 0.5
				mid_y = (traj_array[:, 1].max()+traj_array[:, 1].min()) * 0.5
				self.ax3.set_xlim(mid_x - max_range, mid_x + max_range)
				self.ax3.set_ylim(mid_z - max_range, mid_z + max_range)
				self.ax3.set_zlim(mid_y - max_range, mid_y + max_range)
		
		elif not self.use_3d_subplot and self.ax3 is not None:
			# Zoomed BEV mode - show trajectory and 3D points around current position
			if len(trajectory_points) > 1:
				traj_array = np.array(trajectory_points)
				x = traj_array[:, 0]
				z = traj_array[:, 2]
				
				# Plot full trajectory
				self.ax3.plot(x, z, 'b-', linewidth=2, label='Estimated Path')
				self.ax3.scatter(traj_array[-1, 0], traj_array[-1, 2], c='red', s=150, marker='o', 
							edgecolors='yellow', linewidth=2, zorder=10, label='Current Position')
				self.ax3.scatter(0, 0, c='green', s=150, marker='s', label='Start', zorder=5)
				
				# Plot 3D points in zoomed BEV
				if points_3d is not None and len(points_3d) > 0:
					self.ax3.scatter(points_3d[:, 0], points_3d[:, 2], 
								   c='cyan', s=2, alpha=0.5, label='3D Points', zorder=1)

			self.ax3.set_xlabel('X (unscaled)', fontsize=12)
			self.ax3.set_ylabel('Z (unscaled)', fontsize=12)
			self.ax3.set_title(f'Zoomed BEV - Current Region', 
							  fontsize=14, fontweight='bold')
			self.ax3.grid(True, alpha=0.3)
			self.ax3.legend(loc='upper right')
			
			# Set zoom around current position (after axis equal to ensure it takes effect)
			if len(trajectory_points) > 1:
				zoom_range = 15.0  # meters around current position
				curr_x, curr_y, curr_z = trajectory_points[-1]
				self.ax3.set_xlim(curr_x - zoom_range, curr_x + zoom_range)
				self.ax3.set_ylim(curr_z - zoom_range, curr_z + zoom_range)
				self.ax3.set_aspect('equal', adjustable='box')

		# Use blitting for faster updates
		self.fig.canvas.draw_idle()
		self.fig.canvas.flush_events()
		
		# Save frame to video if enabled
		if self.save_video:
			self._save_frame_to_video()
	
	def _save_frame_to_video(self):
		"""Save the current matplotlib figure to video."""
		try:
			# Force draw completion
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
			
			# Get the RGBA buffer
			buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
			w, h = self.fig.canvas.get_width_height()
			
			# Reshape buffer - the buffer size should match exactly
			try:
				buf = buf.reshape((h, w, 4))
			except ValueError:
				# If reshape fails, calculate actual dimensions
				actual_pixels = len(buf) // 4
				# Try to maintain aspect ratio
				ratio = w / h
				actual_h = int(np.sqrt(actual_pixels / ratio))
				actual_w = actual_pixels // actual_h
				buf = buf[:actual_h * actual_w * 4].reshape((actual_h, actual_w, 4))
				# print(f"Warning: Buffer size mismatch. Adjusted to {actual_w}x{actual_h}")
			
			# Convert RGBA to BGR for OpenCV
			frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
			
			# Initialize video writer on first frame
			if self.video_writer is None:
				height, width = frame_bgr.shape[:2]
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
				print(f"Initialized video writer: {width}x{height} @ {self.fps} fps")
			
			# Write frame
			self.video_writer.write(frame_bgr)
			
		except Exception as e:
			print(f"Warning: Failed to save frame to video: {e}")
	
	def release(self):
		"""Release video writer resources."""
		if self.video_writer is not None:
			self.video_writer.release()
			print(f"\nVideo saved to: {self.output_path}")
			self.video_writer = None