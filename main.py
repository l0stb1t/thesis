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
						return C_RIGHT_ARM_UP_OPEN
					if 15 < vert_angle_right_arm < 90:
						return C_RIGHT_ARM_UP_CLOSED
			elif (l_ear is not None) and shoulders_width and distance(r_wrist,l_ear) < shoulders_width/4:
				# Right hand close to left ear
				return C_RIGHT_HAND_ON_LEFT_EAR
		else:
			# Both hands up
			# Check if both hands are on the ears
			if (r_ear is not None) and (l_ear is not None):
				ear_dist = distance(r_ear,l_ear)
				if distance(r_wrist,r_ear)<ear_dist/3 and distance(l_wrist,l_ear)<ear_dist/3:
					return C_HANDS_ON_EARS
			# Check if boths hands are closed to each other and above ears 
			# (check right hand is above right ear is enough since hands are closed to each other)
			if shoulders_width and (r_ear is not None):
				near_dist = shoulders_width/3
				if r_ear[0] > r_wrist[0] and distance(r_wrist, l_wrist) < near_dist :
					return C_CLOSE_HANDS_UP
	else:
		if left_hand_up:
			# Only left arm up
			if (l_ear is not None) and (l_ear[1]-neck[1])*(l_wrist[1]-neck[1])>0:
				# Left ear and left hand on the same side
				if vert_angle_left_arm:
					if vert_angle_left_arm < -15:
						return C_LEFT_ARM_UP_CLOSED
					if 15 < vert_angle_left_arm < 90:
						return C_LEFT_ARM_UP_OPEN
			elif (r_ear is not None) and shoulders_width and distance(l_wrist,r_ear) < shoulders_width/4:
				# Left hand close to right ear
				return C_LEFT_HAND_ON_RIGHT_EAR
		else:
			# Both wrists under the neck
			if (neck is not None) and (shoulders_width is not None) and (r_wrist is not None) and (l_wrist is not None):
				near_dist = shoulders_width/3
				if distance(r_wrist, neck) < near_dist and distance(l_wrist, neck) < near_dist :
					return C_HANDS_ON_NECK
	return None

def get_skps(kps_coord, kps_score):
	r = (0,)*C_NKP
	min_x = None
	max_x = None
	min_y = None	
	max_y = None
	
	for i in range(C_NKP):
		if kps_score[i] >= C_KP_THRESHOLD:
			r[i] = kps_coord[i]
			x = r[i][0]
			y = r[i][1]
			
			if max_x is None or x > max_x:
				max_x = x
			if min_x is None or x < min_x:
				min_x = x
			if max_y is None or y > max_y:
				max_y = y
			if min_y is None or y < min_y:
				min_y = y
		else:
			r[i] = None
	return r, (min_x, min_y), (max_x, max_y)

def get_pose_box(kps_coord, kps_score):
	min_x = None
	max_x = None
	min_y = None	
	max_y = None
	
	for i in range(C_NKP):
		if kps_score[i] >= C_KP_THRESHOLD:
			x = kps_coord[i][0]
			y = kps_coord[i][1]
			
			if max_x is None or x > max_x:
				max_x = x
			if min_x is None or x < min_x:
				min_x = x
			if max_y is None or y > max_y:
				max_y = y
			if min_y is None or y < min_y:
				min_y = y
	
	return (np.array((min_x, min_y)), np.array((max_x, max_y)))
	
def get_surf():
    global FRAMEBUFFER
    
    frame = np.ctypeslib.as_array(FRAMEBUFFER).copy()
    surf = pygame.surfarray.make_surface(frame)
    return surf

def draw_pose(surf, kps_coord, kps_score, color):
	for i in range(18):
		if kps_score[i] >= C_KP_THRESHOLD:
			pygame.draw.circle(surf, color, kps_coord[i].astype(np.uint32), 3)

def draw_bound_box(surf, kps_coord, kps_score):
	top, bottom = get_pose_box(kps_coord, kps_score)
	pygame.draw.circle(surf, C_RED, top.astype(np.uint32), 3)
	pygame.draw.circle(surf, C_RED, bottom.astype(np.uint32), 3)
	pygame.draw.rect(surf, C_GREEN, (top[0], top[1], bottom[0]-top[0], bottom[1]-top[1]), 3)
	return top, bottom
				
def renderer(lock):
	global RUNNING
	global NPOSES, FRAMEBUFFER, KP_BUFFER, SCORE_BUFFER, POSESCORE_BUFFER

	pygame.init()
	pygame.font.init()
	myfont = pygame.font.SysFont(None, 20)
	
	LABELS = [None,]*10
	
	POSE_TEXTS = [None,] * 9
	POSE_TEXTS[C_RIGHT_ARM_UP_OPEN] 			= myfont.render('RIGHT_ARM_UP_OPEN', False, C_RED, None)
	POSE_TEXTS[C_RIGHT_ARM_UP_CLOSED] 		= myfont.render('RIGHT_ARM_UP_CLOSED', False, C_RED, None)
	POSE_TEXTS[C_RIGHT_HAND_ON_LEFT_EAR] 	= myfont.render('RIGHT_HAND_ON_LEFT_EAR', False, C_RED, None)
	
	POSE_TEXTS[C_LEFT_ARM_UP_OPEN] 			= myfont.render('LEFT_ARM_UP_OPEN', False, C_RED, None)
	POSE_TEXTS[C_LEFT_ARM_UP_CLOSED] 		= myfont.render('LEFT_ARM_UP_CLOSED', False, C_RED, None)
	POSE_TEXTS[C_LEFT_HAND_ON_RIGHT_EAR] 	= myfont.render('C_LEFT_HAND_ON_RIGHT_EAR', False, C_RED, None)
	
	POSE_TEXTS[C_HANDS_ON_EARS] 				= myfont.render('HANDS_ON_EARS', False, C_RED, None)
	POSE_TEXTS[C_CLOSE_HANDS_UP] 				= myfont.render('CLOSE_HANDS_UP', False, C_RED, None)
	POSE_TEXTS[C_HANDS_ON_NECK] 				= myfont.render('HANDS_ON_NECK', False, C_RED, None)
		
	display = pygame.display.set_mode(appsink_size)
	
	frame_count = 0
	start_time = time.time()
	running = True
	while RUNNING:
		frame_count += 1
		
		surf = get_surf()
		lock.acquire()
		nposes = NPOSES.value
		if nposes:
			kps_coords 	= np.ctypeslib.as_array(KP_BUFFER).copy()
			kps_scores 	= np.ctypeslib.as_array(SCORE_BUFFER).copy()
			pose_scores = np.ctypeslib.as_array(POSESCORE_BUFFER).copy()
			lock.release()
			
			for i in range(nposes):
				if pose_scores[i] >= C_PSCORE_THRESHOLD:
					kps_coord = kps_coords[i]
					kps_score = kps_scores[i]
						
					#color = C_COLORS[i % 3]
					draw_pose(surf, kps_coord, kps_score, C_YELLOW)
					
					top, bottom = draw_bound_box(surf, kps_coord, kps_score)
					pose = check_pose(kps_coord, kps_score)
					if pose is not None:
						LABELS[i] = (POSE_TEXTS[pose], (top[1], top[0]))
					else:
						LABELS[i] = None
					'''
					print (pose)
					if pose == C_LEFT_ARM_UP_OPEN or pose == C_LEFT_ARM_UP_CLOSED:
						pygame.draw.circle(surf, C_RED, kps_coord[C_LWRIST].astype(np.uint32), 10)
					if pose == C_RIGHT_ARM_UP_OPEN or pose == C_RIGHT_ARM_UP_CLOSED:
						pygame.draw.circle(surf, C_GREEN, kps_coord[C_RWRIST].astype(np.uint32), 10)
					'''
		else:
			lock.release()
			
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				RUNNING.value = 0
				break
		surf = pygame.transform.rotate(surf, -90)
		surf = pygame.transform.flip(surf, True, False)		
		display.blit(surf, (0, 0))
		for i in range(nposes):
			if LABELS[i]:
				display.blit(LABELS[i][0], LABELS[i][1])
		pygame.display.update()
		pygame.time.delay(15)
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
					pygame.time.delay(30)
def main():
	global RUNNING
	global NPOSES, FRAMEBUFFER, KP_BUFFER, SCORE_BUFFER, POSESCORE_BUFFER

	frame 			= np.zeros((appsink_size[1], appsink_size[0], 3) , dtype=np.uint8)
	t 					= np.ctypeslib.as_ctypes(frame)
	FRAMEBUFFER 	= sharedctypes.RawArray(t._type_, (t))
	
	kp_buffer 		= np.zeros((C_MAXPOSE, C_NKP, 2) , dtype=np.float64)
	t 					= np.ctypeslib.as_ctypes(kp_buffer)
	KP_BUFFER		= sharedctypes.RawArray(t._type_, (t))
	
	score_buffer 	= np.zeros((C_MAXPOSE, C_NKP) , dtype=np.float64)
	t 					= np.ctypeslib.as_ctypes(score_buffer)
	SCORE_BUFFER	= sharedctypes.RawArray(t._type_, (t))
	
	posescore_buffer 	= np.zeros((C_MAXPOSE,) , dtype=np.float64)
	t 						= np.ctypeslib.as_ctypes(posescore_buffer)
	POSESCORE_BUFFER	= sharedctypes.RawArray(t._type_, (t))
	
	NPOSES 			= sharedctypes.RawValue(ctypes.c_ushort)
	RUNNING			= sharedctypes.RawValue(ctypes.c_ubyte, 1)

	lock = mp.Lock()
	p_renderer = mp.Process(target=renderer, args=(lock, ))
	p_pose_worker = mp.Process(target=pose_worker)
	#p_pose_worker.start()
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

			lock.acquire()
			FRAMEBUFFER[:] 					= np.ctypeslib.as_ctypes(input_img)
			if nposes:
				NPOSES.value = nposes
				KP_BUFFER[:nposes] 			= np.ctypeslib.as_ctypes(kps)
				SCORE_BUFFER[:nposes] 		= np.ctypeslib.as_ctypes(kps_score)
				POSESCORE_BUFFER[:] 			= np.ctypeslib.as_ctypes(pose_scores)
			lock.release()

		except:
			traceback.print_exc()
			RUNNING.value = 0
			break

	end_time = time.time()
	print ('Processing FPS:', frame_count/(end_time - start_time))

if __name__ == "__main__":
	main()
