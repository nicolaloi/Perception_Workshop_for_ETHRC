import numpy as np
import cv2

K_MIN_NUM_FEATURE = 1500
MIN_BASELINE = 0.3  # Minimum baseline for triangulation (normalized units)
MIN_TRIANGULATION_ANGLE = 1.0  # Minimum angle in degrees for triangulation

# TODO: implement the function to track features between two frames
# have a look at cv2.calcOpticalFlowPyrLK and play with its parameters!
def TODO_feature_tracking(prev_image, curr_image, prev_features):
	random_kp2 = prev_features + np.random.randint(-5, 5, prev_features.shape).astype(np.float32)
	random_status = np.ones((prev_features.shape[0],1), dtype=np.uint8)

	return random_kp2, random_status

# TODO: implement the function to detect features in the image
# have a look at cv2.goodFeaturesToTrack and play with its parameters!
def TODO_detect_features(img):
	random_keypoints_v = np.random.randint(0, img.shape[0], int(K_MIN_NUM_FEATURE / 3)).astype(np.float32)
	random_keypoints_u = np.random.randint(0, img.shape[1], int(K_MIN_NUM_FEATURE / 3)).astype(np.float32)
	random_new_current_features = np.stack((random_keypoints_u, random_keypoints_v), axis=-1)

	return random_new_current_features

# TODO: implement the function to find the essential matrix between two sets of points
# have a look at cv2.findEssentialMat and play with its RANSAC parameters!
def TODO_find_essential_matrix(prev_points, curr_points, camera_matrix):
	# use cv2.findEssentialMat to compute the essential matrix
	# check https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gab705726dc6b655acf50bc936942824ef
	random_E = np.eye(3)
	random_mask = np.ones((prev_points.shape[0], 1), dtype=np.uint8)

	return random_E, random_mask


class VisualOdometry:
	def __init__(self, K):
		self.K = K
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.last_features = None
		self.new_features = None
		self.inliers = None

		self.max_point_distance = 100.0  # Maximum distance for 3D point filtering
		
		# 3D point triangulation
		self.points_3d = []  # Store triangulated 3D points
		self.frame_idx = 0
		self.prev_R = None
		self.prev_t = None
		self.prev_inlier_points = None

	def processFrame(self):
		if self.last_features is None:
			# first frame, just detect first keypoints
			self.last_features = TODO_detect_features(self.new_frame)
			self.cur_t = np.zeros((3,1))
			self.cur_R = np.eye(3)
			self.frame_idx += 1
			return
		
		# Store previous pose for triangulation
		self.prev_R = self.cur_R.copy()
		self.prev_t = self.cur_t.copy()
				
		kp2, st = TODO_feature_tracking(self.last_frame, self.new_frame, self.last_features)
		st = st.reshape(st.shape[0])
		self.last_features = self.last_features[st == 1]
		self.new_features = kp2[st == 1]
		E, mask = TODO_find_essential_matrix(self.last_features, self.new_features, self.K)
		pt1 = self.last_features[mask.ravel() == 1]
		pt2 = self.new_features[mask.ravel() == 1]
		pt_inliers = mask.ravel()
		_, R, t, mask = cv2.recoverPose(E, pt2, pt1, cameraMatrix=self.K)
		# Create a full-size inliers mask
		self.inliers = np.zeros(len(self.new_features), dtype=np.uint8)
		# Mark the points that passed both essential matrix RANSAC and pose recovery
		inlier_indices = np.where(mask.ravel() == 255)[0]
		essential_inlier_indices = np.where(pt_inliers == 1)[0]
		final_inlier_indices = essential_inlier_indices[inlier_indices]
		self.inliers[final_inlier_indices] = 1
		
		# Store previous inlier points for triangulation
		self.prev_inlier_points = self.last_features[self.inliers == 1]

		self.cur_t = self.cur_t + self.cur_R.dot(t)
		self.cur_R = R.dot(self.cur_R)
		
		# Triangulate 3D points
		self.triangulate_points()
		
		if len(self.new_features) < K_MIN_NUM_FEATURE:
			new_current_features = TODO_detect_features(self.new_frame)
			self.new_features = np.vstack((self.new_features, new_current_features))
		
		self.frame_idx += 1

	def update(self, img):
		if self.new_features is not None:
			self.last_features = self.new_features
		self.last_frame = self.new_frame.copy() if self.new_frame is not None else None
		self.new_frame = img
		
		self.processFrame()

	def triangulate_points(self):
		"""
		Triangulate 3D points from 2D correspondences between consecutive frames.
		Uses inlier points from the essential matrix decomposition.
		"""
		if self.prev_R is None or self.prev_t is None or self.prev_inlier_points is None:
			return
		
		# Get current inlier points
		cur_inlier_points = self.new_features[self.inliers == 1]
		
		if len(cur_inlier_points) == 0 or len(self.prev_inlier_points) == 0:
			return
		
		# Construct projection matrices
		# Previous camera pose (world to camera)
		P1 = self.K @ np.hstack((self.prev_R.T, -self.prev_R.T @ self.prev_t))
		
		# Current camera pose (world to camera)
		P2 = self.K @ np.hstack((self.cur_R.T, -self.cur_R.T @ self.cur_t))
		
		# Triangulate points
		# cv2.triangulatePoints expects points in format (2, N)
		points_4d_hom = cv2.triangulatePoints(P1, P2, 
											   self.prev_inlier_points.T, 
											   cur_inlier_points.T)
		
		# Convert from homogeneous coordinates to 3D
		points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
		points_3d = points_3d.T  # Shape: (N, 3)
		
		# Filter points based on depth (should be in front of both cameras)
		valid_points = []
		for i, pt_3d in enumerate(points_3d):
			# Check if point is in front of both cameras
			# Transform to previous camera frame
			pt_prev_cam = self.prev_R.T @ (pt_3d.reshape(3, 1) - self.prev_t)
			# Transform to current camera frame
			pt_cur_cam = self.cur_R.T @ (pt_3d.reshape(3, 1) - self.cur_t)
			
			# Check depth (z-coordinate should be positive in camera frame)
			if pt_prev_cam[2] > 0 and pt_cur_cam[2] > 0:
				# Also filter out points that are too far (likely outliers or anyway very noisy)
				depth = np.linalg.norm(pt_3d - self.cur_t.flatten())
				if depth < self.max_point_distance:  # Arbitrary threshold
					valid_points.append(pt_3d)

		# Filter points too close to each other based on triangulation angle
		filtered_points = []
		for i, pt_3d in enumerate(valid_points):
			# Vectors from camera centers to the 3D point
			vec_prev = pt_3d.reshape(3, 1) - self.prev_t
			vec_cur = pt_3d.reshape(3, 1) - self.cur_t
			
			# Normalize vectors
			vec_prev_norm = vec_prev / np.linalg.norm(vec_prev)
			vec_cur_norm = vec_cur / np.linalg.norm(vec_cur)
			
			# Compute angle between the two vectors
			cos_angle = np.clip(np.dot(vec_prev_norm.T, vec_cur_norm), -1.0, 1.0)
			angle = np.arccos(cos_angle) * (180.0 / np.pi)  # Convert to degrees
			
			if angle >= MIN_TRIANGULATION_ANGLE:
				filtered_points.append(pt_3d)

		# Filter points lower than the camera (filter out ground points)
		filtered_points_2 = []
		for i, pt_3d in enumerate(filtered_points):
			if pt_3d[1] > self.cur_t[1]:  # y-coordinate greater than camera y (y is up)
				filtered_points_2.append(pt_3d)
			
		# Store valid 3D points
		if len(filtered_points_2) > 0:
			self.points_3d.extend(filtered_points_2)
	
	def get_recent_3d_points(self, max_points=500, skip_interval=10):
		"""
		Get the most recent 3D points for visualization.
		Limits the number of points to avoid cluttering the visualization.
		"""
		return np.array(self.points_3d[-max_points::skip_interval])