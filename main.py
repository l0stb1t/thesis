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
from math import atan2, degrees, sqrt, pi, floor
from constants import *
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='.tflite model path.', required=False)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
parser.add_argument('--res', help='Resolution', default='480x360', choices=['480x360', '640x480', '1280x720'])
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--ifile', type=str, default=None, help="Optionally use an image file instead of a live camera")
parser.add_argument('--interact', type=int, default=0)


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

def get_frame():
	return np.ctypeslib.as_array(FRAMEBUFFER).copy()
	
def get_surf():
    global FRAMEBUFFER
    
    frame = np.ctypeslib.as_array(FRAMEBUFFER).copy()
    surf = pygame.surfarray.make_surface(frame)
    return surf, frame

def get_pose_shared_data(lock):
	global NPOSES, FRAMEBUFFER, KP_BUFFER, KP_SCORE_BUFFER, POSESCORE_BUFFER
	
	lock.acquire()
	nposes = NPOSES.value
	if nposes:
		kps_coords 	= np.ctypeslib.as_array(KP_BUFFER).copy()
		kps_scores 	= np.ctypeslib.as_array(KP_SCORE_BUFFER).copy()
		pose_scores = np.ctypeslib.as_array(POSESCORE_BUFFER).copy()
	else:
		kps_coords = None
		kps_scores = None
		pose_scores = None
	lock.release()
	return nposes, kps_coords, kps_scores, pose_scores
	
def renderer_tracker(lock, mp_event):
	global args, RUNNING
	
	pygame.init()
	display = pygame.display.set_mode(appsink_size)
	pygame.display.set_caption('tracker')
	
	features 		= [None]*C_NTRACK
	feature_count 	= 0
	colors			= [C_YELLOW, C_GREEN] + [rand_color() for i in range(C_NTRACK-2)]
	
	frame_count = 0
	start_time = time.time()
	while RUNNING:
		if args.interact:
			mp_event.wait()
			mp_event.clear()
		
		frame_count += 1
		surf, frame = get_surf()
		nposes, kps_coords, kps_scores, pose_scores = get_pose_shared_data(lock)
		
		pose_idxs = []
		for i in range(nposes): # only take pose with score > C_PSCORE_THRESHOLD
			if pose_scores[i] >= C_PSCORE_THRESHOLD:
				pose_idxs.append(i)
		nposes = len(pose_idxs)
		if nposes: # there are people in the frame
			sort = list(np.argsort(pose_scores)[::-1][:nposes])
			print (nposes)
			print (sort)
			current_frame 	= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert current frame to gray scale
			if feature_count == 0 or ( (feature_count<C_NTRACK) and (feature_count<nposes) ): # we are currently tracking no one
				feature_count = 0
				for i in range(0, min(nposes, C_NTRACK)): # extract new feature of C_NTRACK most highest score pose
					kps_coord 		= kps_coords[sort[i]]
					kps_score 		= kps_scores[sort[i]]
					#top, bottom 	= draw_bound_box(surf, kps_coord, kps_score, C_BLUE)
					top,  bottom	= get_bound_box(kps_coord, kps_score)
					
					mask = np.zeros(frame.shape[:-1], dtype=np.uint8) # init mask
					mask[top[0]:bottom[0], top[1]:bottom[1]] = 1
					features[i] = cv2.goodFeaturesToTrack(current_frame, mask=mask, **feature_params)# extract features
					print ('goodFeaturesToTrack', features[i].shape) 
					feature_count += 1
					
			else: # there are  >= 1 features
				print ('feature_count:', feature_count)
				for i in range(feature_count):
					if features[i] is None:
						continue
					old_features = features[i] # we gonna findout which pose matchs old feature
					
					found = 0
					for pose_idx in sort:
						kps_coord 		= kps_coords[pose_idx]
						kps_score 		= kps_scores[pose_idx]
						top, bottom 	= draw_bound_box(surf, kps_coord, kps_score, C_BLUE)
						#top,  bottom	= get_bound_box(kps_coord, kps_score)
						
						mask = np.zeros(frame.shape[:-1], dtype=np.uint8) # init mask
						mask[top[0]:bottom[0], top[1]:bottom[1]] = 1 
						
						new_features, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, old_features, None, **lk_params) # extract new features
						new_features = new_features[st==1]
						
						found_features = [] # only get feature inside current pose bounding box
						for e in new_features:
							try:
								if mask[floor(e[1])][floor(e[0])] == 1: # feature is in float
									found_features.append(e)
							except:
								pass
						found_features = np.array(found_features, dtype=np.float32)
													
						if found_features.shape[0] >= C_FEATURE_THRESHOLD: # if there are more than C_FEATURE_THRESHOLD consider a valid
								found = 1
								draw_points(surf, found_features, color=colors[i]) # for debugging
								draw_bound_box(surf, kps_coord, kps_score, colors[i]) #draw bound box
								sort.remove(pose_idx) #remove current pose from list
								
								if found_features.shape[0] <= C_FEATURE_THRESHOLD2: #our feature count is too low
									features[i] = cv2.goodFeaturesToTrack(current_frame, mask=mask, **feature_params)# we reupdate our feature
									print ('reupdate feature', features[i].shape)
								else:
									features[i] = found_features.reshape(-1, 1, 2) #update new features
									print ('new feature', found_features.shape)
								break
								
					if not found: # we don't have a match
						features = [None]*C_NTRACK
						feature_count = 0
						
			prev_frame = current_frame

		surf = pygame.transform.rotate(surf, -90)
		surf = pygame.transform.flip(surf, True, False)
		display.blit(surf, (0, 0))
		pygame.display.update()
		pygame.time.delay(15)
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				RUNNING.value = 0
				break
				
	end_time = time.time()
	print ('Tracker FPS:', frame_count/(end_time - start_time))
	pygame.quit()

def renderer_tracker2(lock, mp_event):
	pass    

def renderer_poser(lock, mp_event):
	global RUNNING, PAUSE
	global NPOSES, FRAMEBUFFER, KP_BUFFER, KP_SCORE_BUFFER, POSESCORE_BUFFER

	pygame.init()
	display = pygame.display.set_mode(appsink_size)
	pygame.display.set_caption('poser')

	frame_count = 0
	start_time = time.time()
	while RUNNING:			
		if not PAUSE:
			frame_count += 1
			surf, frame = get_surf()
			lock.acquire()
			nposes = NPOSES.value
			if nposes:
				kps_coords 	= np.ctypeslib.as_array(KP_BUFFER).copy()
				kps_scores 	= np.ctypeslib.as_array(KP_SCORE_BUFFER).copy()
				pose_scores = np.ctypeslib.as_array(POSESCORE_BUFFER).copy()
				lock.release()
				
				for i in range(nposes):
					if pose_scores[i] >= C_PSCORE_THRESHOLD:
						kps_coord = kps_coords[i]
						kps_score = kps_scores[i]
							
						color = rand_color()
						draw_pose(surf, kps_coord, kps_score, color)
						top, bottom = draw_bound_box(surf, kps_coord, kps_score, color)
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
				
			surf = pygame.transform.rotate(surf, -90)
			surf = pygame.transform.flip(surf, True, False)		
			display.blit(surf, (0, 0))
			for i in range(nposes):
				if LABELS[i]:
					display.blit(LABELS[i][0], LABELS[i][1])
			pygame.display.update()
			pygame.time.delay(15)
			
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				RUNNING.value = 0
				break
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_p:
					PAUSE.value ^= 1
					
	end_time = time.time()
	print ('Poser FPS:', frame_count/(end_time - start_time))
	pygame.quit()

def main():
	global RUNNING, PAUSE
	global NPOSES, FRAMEBUFFER, KP_BUFFER, KP_SCORE_BUFFER, POSESCORE_BUFFER

	FRAMEBUFFER 		= sharedctypes.RawValue(ctypes.c_ubyte*3*appsink_size[0]*appsink_size[1])
	KP_BUFFER			= sharedctypes.RawValue(ctypes.c_uint*2*C_NKP*C_MAXPOSE)
	KP_SCORE_BUFFER		= sharedctypes.RawValue(ctypes.c_double*C_NKP*C_MAXPOSE)
	POSESCORE_BUFFER	= sharedctypes.RawValue(ctypes.c_double*C_MAXPOSE)
	HEIGHT_BUFFER		= sharedctypes.RawValue(ctypes.c_uint*C_MAXPOSE)
	HEIGHT_IDX_BUFFER	= sharedctypes.RawValue(ctypes.c_uint*C_MAXPOSE)
	
	NPOSES 				= sharedctypes.RawValue(ctypes.c_ushort)
	RUNNING				= sharedctypes.RawValue(ctypes.c_ubyte, 1)
	PAUSE				= sharedctypes.RawValue(ctypes.c_ubyte, 0)

	lock 		= mp.Lock()
	mp_event 	= mp.Event()
	
	p_renderer_poser 	= mp.Process(target=renderer_poser, args=(lock, mp_event))
	p_renderer_tracker 	= mp.Process(target=renderer_tracker, args=(lock, mp_event))
	# p_renderer_poser.start()
	p_renderer_tracker.start()

	if args.file is not None:
		cap = cv2.VideoCapture(args.file)
	else:
		cap = cv2.VideoCapture(args.cam_id)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, src_size[0])
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, src_size[1])

	
	frame_count = 0
	start_time = time.time()
	pose_height = np.zeros(C_MAXPOSE, dtype=np.uint32)
	while RUNNING:
		if not PAUSE:			
			try:
				frame_count += 1
				cap_res, cap_frame 	= cap.read()
				input_img 			= cv2.resize(cap_frame, appsink_size, cv2.INTER_NEAREST)
				input_img 			= cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
				nposes, pose_scores, kps, kps_score = engine.DetectPosesInImage(input_img)

				lock.acquire()
				FRAMEBUFFER[:] 					= np.ctypeslib.as_ctypes(input_img)
				if nposes:						
					NPOSES.value 				= nposes
					KP_BUFFER[:nposes] 			= np.ctypeslib.as_ctypes(kps.astype(np.uint32))
					KP_SCORE_BUFFER[:nposes] 	= np.ctypeslib.as_ctypes(kps_score)
					POSESCORE_BUFFER[:] 		= np.ctypeslib.as_ctypes(pose_scores)
					
					for i in range(nposes):
						top, bottom = get_bound_box(KP_BUFFER[i], KP_SCORE_BUFFER[i])
						pose_height[i] = bottom[1]-top[1]
						
				lock.release()
				
				if args.interact:
					input('')
					mp_event.set()
					
			except:
				traceback.print_exc()
				RUNNING.value = 0
				break

	end_time = time.time()
	print ('Posenet FPS:', frame_count/(end_time - start_time))

if __name__ == "__main__":
	main()
