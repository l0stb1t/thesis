import cv2, sys
import numpy as np
sys.path.append('..')
from util import *
from constants import *
from pose_engine import PoseEngine

class PN:
	def __init__(self):
		model = '../models/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite'
		self.engine = PoseEngine(model, mirror=False)
		
	def eval(self, frame):
		nposes, pose_scores, kps_coords, kps_scores = self.engine.DetectPosesInImage(frame)
		kps_coords = kps_coords.astype(np.int32)
		
		pose_list = []
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
