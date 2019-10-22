import sys
sys.path.insert(0, '/home/bit/project-posenet/')

import cv2
import time
import ctypes
import pygame
import argparse
import traceback
import numpy as np
import multiprocessing as mp
from pose_engine import PoseEngine
from multiprocessing import sharedctypes
from math import atan2, degrees, sqrt, pi
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='.tflite model path.', required=False)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
parser.add_argument('--res', help='Resolution', default='480x360', choices=['480x360', '640x480', '1280x720'])
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--ifile', type=str, default=None, help="Optionally use an image file instead of a live camera")
args = parser.parse_args()

default_model = 'models/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
if args.res == '480x360':
	src_size = (640, 480)
	appsink_size = (480, 360)
	model = args.model or default_model % (353, 481)
elif args.res == '640x480':
	src_size = (640, 480)
	appsink_size = (640, 480)
	model = args.model or default_model % (481, 641)
elif args.res == '1280x720':
	src_size = (1280, 720)
	appsink_size = (1280, 720)
	model = args.model or default_model % (721, 1281)

print('Loading model: ', model)
engine = PoseEngine(model, mirror=args.mirror)

def vertical_angle(A, B):
	if A is None or B is None:
		return None
	return degrees(atan2(B[0]-A[0],B[1]-A[1]) - pi/2)

def distance (A, B):
	return np.linalg.norm(A-B)

def check_pose(keypoint_coord, keypoint_score):
	if keypoint_score[-1] > C_KP_THRESHOLD:
		neck = keypoint_coord[-1]
	else:
		neck = None

	if keypoint_score[10] > C_KP_THRESHOLD:
		r_wrist = keypoint_coord[10]
	else:
		r_wrist = None

	if keypoint_score[9] > C_KP_THRESHOLD:
		l_wrist = keypoint_coord[9]
	else:
		l_wrist = None

	if keypoint_score[8] > C_KP_THRESHOLD:
		r_elbow = keypoint_coord[8]
	else:
		r_elbow = None

	if keypoint_score[7] > C_KP_THRESHOLD:
		l_elbow = keypoint_coord[7]
	else:
		l_elbow = None

	if keypoint_score[6] > C_KP_THRESHOLD:
		r_shoulder = keypoint_coord[6]
	else:
		r_shoulder = None

	if keypoint_score[5] > C_KP_THRESHOLD:
		l_shoulder = keypoint_coord[5]
	else:
		l_shoulder = None

	if keypoint_score[4] > C_KP_THRESHOLD:
		r_ear = keypoint_coord[4]
	else:
		r_ear = None

	if keypoint_score[3] > C_KP_THRESHOLD:
		l_ear = keypoint_coord[3]
	else:
		l_ear = None

	shoulders_width = distance(r_shoulder,l_shoulder) if (r_shoulder is not None) and (l_shoulder is not None) else None

	vert_angle_right_arm = vertical_angle(r_wrist, r_elbow)
	vert_angle_left_arm = vertical_angle(l_wrist, l_elbow)
	
	left_hand_up = (neck is not None) and (l_wrist is not None) and l_wrist[0] < neck[0]
	right_hand_up = (neck is not None) and (r_wrist is not None)  and r_wrist[0] < neck[0]
	#print (left_hand_up, right_hand_up)
	if right_hand_up:
		if not left_hand_up:
			# Only right arm up
			if (r_ear is not None) and (r_ear[1]-neck[1])*(r_wrist[1]-neck[1])>0:
			# Right ear and right hand on the same side
				if vert_angle_right_arm:
					if vert_angle_right_arm < -15:
						return "RIGHT_ARM_UP_OPEN"
					if 15 < vert_angle_right_arm < 90:
						return "RIGHT_ARM_UP_CLOSED"
			elif (l_ear is not None) and shoulders_width and distance(r_wrist,l_ear) < shoulders_width/4:
				# Right hand close to left ear
				return "RIGHT_HAND_ON_LEFT_EAR"
		else:
			# Both hands up
			# Check if both hands are on the ears
			if (r_ear is not None) and (l_ear is not None):
				ear_dist = distance(r_ear,l_ear)
				if distance(r_wrist,r_ear)<ear_dist/3 and distance(l_wrist,l_ear)<ear_dist/3:
					return("HANDS_ON_EARS")
			# Check if boths hands are closed to each other and above ears 
			# (check right hand is above right ear is enough since hands are closed to each other)
			if shoulders_width and (r_ear is not None):
				near_dist = shoulders_width/3
				if r_ear[0] > r_wrist[0] and distance(r_wrist, l_wrist) < near_dist :
					return "CLOSE_HANDS_UP"

	else:
		if left_hand_up:
			# Only left arm up
			if (l_ear is not None) and (l_ear[1]-neck[1])*(l_wrist[1]-neck[1])>0:
				# Left ear and left hand on the same side
				if vert_angle_left_arm:
					if vert_angle_left_arm < -15:
						return "LEFT_ARM_UP_CLOSED"
					if 15 < vert_angle_left_arm < 90:
						return "LEFT_ARM_UP_OPEN"
			elif (r_ear is not None) and shoulders_width and distance(l_wrist,r_ear) < shoulders_width/4:
				# Left hand close to right ear
				return "LEFT_HAND_ON_RIGHT_EAR"
		else:
			# Both wrists under the neck
			if (neck is not None) and (shoulders_width is not None) and (r_wrist is not None) and (l_wrist is not None):
				near_dist = shoulders_width/3
				if distance(r_wrist, neck) < near_dist and distance(l_wrist, neck) < near_dist :
					return "HANDS_ON_NECK"

	return None


def get_frame():
    global FRAMEBUFFER
    
    frame = np.ctypeslib.as_array(FRAMEBUFFER).copy()
    frame = np.rot90(frame)
    return frame

def round_kp(kp):
	return kp.astype(np.uint8)

def draw_pose(surf, kps_coord, kps_score):
	for i in range(18):
		if kps_score[i] >= C_KP_THRESHOLD:
			pygame.draw.circle(surf, C_RED, round_kp(kps_coord[i]), 3)
	return surf

def renderer():
	global RUNNING
	global NPOSES, FRAMEBUFFER, KP_BUFFER, SCORE_BUFFER, POSESCORE_BUFFER

	pygame.init()
	display = pygame.display.set_mode(appsink_size)
	
	frame_count = 0
	start_time = time.time()
	running = True
	while RUNNING:
		frame_count += 1
		
		frame = get_frame()
		surf = pygame.surfarray.make_surface(frame)
		
		nposes = NPOSES.value
		if nposes:
			kps_coords 	= np.ctypeslib.as_array(KP_BUFFER).copy()
			kps_scores 	= np.ctypeslib.as_array(SCORE_BUFFER).copy()
			pose_scores = np.ctypeslib.as_array(POSESCORE_BUFFER).copy()
			
			for i in range(nposes):
				if pose_scores[i] >= C_PSCORE_THRESHOLD:
					kps_coord = kps_coords[i]
					kps_score = kps_scores[i]
					#pygame.draw.circle(surf, C_RED, (10, 10), 3)
					#draw_pose(surf, kps_coord, kps_score)
					
			
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				RUNNING.value = 0
				break
		display.blit(surf, (0, 0))
		pygame.display.update()
	end_time = time.time()
	print ('Rendering FPS:', frame_count/(end_time - start_time))
	pygame.quit()

def pose_worker():
	global RUNNING
	global NPOSES, FRAMEBUFFER, KP_BUFFER, SCORE_BUFFER, POSESCORE_BUFFER
	
	while RUNNING:
		nposes = NPOSES.value
		if nposes:
			kps_coords 	= np.ctypeslib.as_array(KP_BUFFER)
			kps_scores 	= np.ctypeslib.as_array(SCORE_BUFFER)
			pose_scores = np.ctypeslib.as_array(POSESCORE_BUFFER)
		
			for i in range(nposes):
				if pose_scores[i] >= C_PSCORE_THRESHOLD:
					kps_coords 	= np.ctypeslib.as_array(KP_BUFFER).copy()
					kps_scores 	= np.ctypeslib.as_array(SCORE_BUFFER).copy()
					pose_scores = np.ctypeslib.as_array(POSESCORE_BUFFER).copy()
					
					t = check_pose(kps_coords[i], kps_scores[i])
					if t:
						print (t)
					
def main():
	global RUNNING
	global NPOSES, FRAMEBUFFER, KP_BUFFER, SCORE_BUFFER, POSESCORE_BUFFER

	frame 			= np.zeros((appsink_size[1], appsink_size[0], 3) , dtype=np.uint8)
	t 				= np.ctypeslib.as_ctypes(frame)
	FRAMEBUFFER 	= sharedctypes.RawArray(t._type_, (t))
	
	kp_buffer 		= np.zeros((C_MAXPOSE, C_NKP, 2) , dtype=np.float64)
	t 				= np.ctypeslib.as_ctypes(kp_buffer)
	KP_BUFFER		= sharedctypes.RawArray(t._type_, (t))
	
	score_buffer 	= np.zeros((C_MAXPOSE, C_NKP) , dtype=np.float64)
	t 				= np.ctypeslib.as_ctypes(score_buffer)
	SCORE_BUFFER	= sharedctypes.RawArray(t._type_, (t))
	
	posescore_buffer 	= np.zeros((C_MAXPOSE,) , dtype=np.float64)
	t 					= np.ctypeslib.as_ctypes(posescore_buffer)
	POSESCORE_BUFFER	= sharedctypes.RawArray(t._type_, (t))
	
	NPOSES 			= sharedctypes.RawValue(ctypes.c_ushort)
	RUNNING			= sharedctypes.RawValue(ctypes.c_ubyte, 1)

	p_renderer = mp.Process(target=renderer)
	p_pose_worker = mp.Process(target=pose_worker)
	p_pose_worker.start()
	p_renderer.start()
	#signal child process to start

	if args.file is not None:
		cap = cv2.VideoCapture(args.file)
	else:
		cap = cv2.VideoCapture(args.cam_id)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, src_size[0])
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, src_size[1])

	frame_count = 0
	start_time = time.time()
	while RUNNING:
		try:
			frame_count += 1
			cap_res, cap_frame = cap.read()
			input_img = cv2.resize(cap_frame, appsink_size, cv2.INTER_NEAREST)
			input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
			nposes, pose_scores, kps, kps_score = engine.DetectPosesInImage(input_img)

			# print (kps)
			FRAMEBUFFER[:] 					= np.ctypeslib.as_ctypes(input_img)
			if nposes:
				NPOSES.value = nposes
				KP_BUFFER[:nposes] 			= np.ctypeslib.as_ctypes(kps)
				SCORE_BUFFER[:nposes] 		= np.ctypeslib.as_ctypes(kps_score)
				POSESCORE_BUFFER[:] 		= np.ctypeslib.as_ctypes(pose_scores)

		except:
			traceback.print_exc()
			RUNNING.value = 0
			break

	end_time = time.time()
	print ('Processing FPS:', frame_count/(end_time - start_time))

if __name__ == "__main__":
	main()
