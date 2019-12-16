import cv2, sys
sys.path.append('..')
from util import *
from constants import *
import numpy as np
from pose_engine import PoseEngine

class PN:
	def __init__(self):
		model = '../models/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite'
		self.engine = PoseEngine(model, mirror=False)
	
	'''
	def eval(self, frame):
		nposes, pose_scores, kps_coords, kps_scores = self.engine.DetectPosesInImage(frame)
		if nposes:
			max_idx = np.argmax(pose_scores[:nposes])
			self.kps_coord = kps_coords[max_idx]
			self.kps_score = kps_scores[max_idx]
			
			self.shoulders_width = np.linalg.norm(self.kps_coord[C_LSHOULDER] - self.kps_coord[C_RSHOULDER])
		return nposes
	'''
		
	def eval(self, frame):
		pose_list = []
		nposes, pose_scores, kps_coords, kps_scores = self.engine.DetectPosesInImage(frame)
		kps_coords = kps_coords.astype(np.int32)
		if nposes:
			for i in range(nposes):
				if pose_scores[i] >= C_PSCORE_THRESHOLD:
					kps_coord = kps_coords[i]
					kps_score = kps_scores[i]
					
					keypoints = {}
					for j in range(C_NKP):
						if kps_score[j] >= C_KP_THRESHOLD:
							keypoints[C_KP_NAMES[j]] = Keypoint(C_KP_NAMES[j], kps_coord[j], kps_score[j])
					pose_list.append(Pose(keypoints, pose_scores[i]))
		return pose_list
		
	def get_body_kp(self, kp_id):
		if self.kps_score[kp_id] >= C_KP_THRESHOLD:
			return (int(self.kps_coord[kp_id][1]), int(self.kps_coord[kp_id][0]))
		else:
			return None
		
	def draw_body(self, frame):
		keypoints = []
		for i in range(C_NKP):
			kp_coord = self.kps_coord[i]
			if self.kps_score[i] >= C_KP_THRESHOLD:
				keypoints.append(cv2.KeyPoint(kp_coord[1], kp_coord[0], 5))
		cv2.drawKeypoints(frame, keypoints, outImage=frame, color = C_BLUE)
			
