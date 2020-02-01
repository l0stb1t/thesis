import sys
sys.path.insert(0, '/home/bit/project-posenet/')

import pdb
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
from constants import *
from util import *

from edgetpu.detection.engine import DetectionEngine
from PIL import Image

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
pose_engine = PoseEngine(model, mirror=args.mirror)
face_engine = DetectionEngine('/home/pi/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')

def get_surf(lock):
    global FRAMEBUFFER
    
    lock.acquire()
    frame = np.ctypeslib.as_array(FRAMEBUFFER).copy()
    lock.release()
    ''' Pygame use (x, y) frame format so have to convert it but return the original frame for later use with opencv '''
    surf = pygame.surfarray.make_surface(np.rot90(frame)[::-1])
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
	
def get_pose_shared_data2(lock):
	global NPOSES, FRAMEBUFFER, KP_BUFFER, KP_SCORE_BUFFER, POSESCORE_BUFFER
	
	pose_list = []
	
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
				
				keypoints = {}
				for j in range(C_NKP):
					if kps_score[j] >= C_KP_THRESHOLD:
						keypoints[C_KP_NAMES[j]] = Keypoint(C_KP_NAMES[j], kps_coord[j], kps_score[j])
				pose_list.append(Pose(keypoints, pose_scores[i]))
	else:
		lock.release()
	return pose_list
	
def renderer_tracker(lock, mp_event):
	global args, RUNNING
	
	display = init_pygame_window('tracking-optical flow')
	
	features 		= [None]*C_NTRACK
	feature_count 	= 0
	colors			= [rand_color() for i in range(C_NTRACK)]
	
	frame_count = 0
	start_time = time.time()
	while RUNNING:
		if args.interact:
			mp_event.wait()
			mp_event.clear()
		
		frame_count += 1
		surf, frame = get_surf(lock)
		nposes, kps_coords, kps_scores, pose_scores = get_pose_shared_data(lock)
		
		count = 0
		if nposes:
			pose_heights 	= np.zeros(C_MAXPOSE, dtype=np.uint32)
			for i in range(nposes): # only take pose with score > C_PSCORE_THRESHOLD
				if pose_scores[i] >= C_PSCORE_THRESHOLD:
					top, bottom = get_boundbox(kps_coords[i], kps_scores[i])
					height = bottom[1] - top[1]
					pose_heights[i] = height
					count += 1
				else:
					pose_heights[i] = 0
			pose_height_idx_sorted = np.argsort(pose_heights)[::-1]
			# print (pose_height_idx_sorted)
			pose_idxs = list(pose_height_idx_sorted[:nposes])
			nposes = count
		if nposes: # there are people in the frame
			sort = pose_idxs
			current_frame 	= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert current frame to gray scale
			if feature_count == 0 or ( (feature_count<C_NTRACK) and (feature_count<nposes) ): # we are currently tracking no one
				feature_count = 0
				for i in range(0, min(nposes, C_NTRACK)): # extract new feature of C_NTRACK most highest score pose
					kps_coord 		= kps_coords[sort[i]]
					kps_score 		= kps_scores[sort[i]]
					#top, bottom 	= draw_boundbox(surf, kps_coord, kps_score, C_BLUE)
					top,  bottom	= get_boundbox(kps_coord, kps_score)
					mask = np.zeros(frame.shape[:-1], dtype=np.uint8) # init mask, mask have the same shape as the frame (y, x)
					mask[top[1]:bottom[1], top[0]:bottom[0]] = 1
					f = cv2.goodFeaturesToTrack(current_frame, mask=mask, **feature_params) # extract features
					if f is not None:
						features[i] = f
						# print ('goodFeaturesToTrack', features[i].shape) 
						feature_count += 1
					
			else: # there are  >= 1 features
				# print ('feature_count:', feature_count)
				for i in range(feature_count):
					if features[i] is None:
						continue
					old_features = features[i] # we gonna findout which pose matchs old feature
					
					found = 0
					for pose_idx in sort:
						kps_coord 		= kps_coords[pose_idx]
						kps_score 		= kps_scores[pose_idx]
						#top, bottom 	= draw_boundbox(surf, kps_coord, kps_score, C_BLUE)
						top,  bottom	= get_boundbox(kps_coord, kps_score)

						mask = np.zeros(frame.shape[:-1], dtype=np.uint8) # init mask, mask have the same shape as the frame (y, x)
						mask[top[1]:bottom[1], top[0]:bottom[0]] = 1 
						
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
								draw_points(surf, found_features, color=C_RED) # for debugging
								#draw_bound_box(surf, kps_coord, kps_score, colors[i]) #draw bound box
								draw_pose2(surf, kps_coord, kps_score, colors[i])
								sort.remove(pose_idx) #remove current pose from list
								
								if found_features.shape[0] <= C_FEATURE_THRESHOLD2: #our feature count is too low
									f = cv2.goodFeaturesToTrack(current_frame, mask=mask, **feature_params)# we reupdate our feature
									if f is not None:
										features[i] = f
										# print ('reupdate feature', features[i].shape)
								else:
									features[i] = found_features.reshape(-1, 1, 2) #update new features
									# print ('new feature', found_features.shape)
								break
								
					if not found: # we don't have a match
						features[i] = None
						feature_count -= 1
						draw_pose2(surf, kps_coord, kps_score, color=C_BLUE)
						
			prev_frame = current_frame
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
	global args, RUNNING
	
	display = init_pygame_window('tracking-distance')
		
	frame_count = 0
	tracker = Tracker()
	start_time = time.time()
	while RUNNING:
		if args.interact:
			mp_event.wait()
			mp_event.clear()
		
		frame_count += 1
		surf, frame = get_surf(lock)
		poses = get_pose_shared_data2(lock)
		
		tracker.feed(poses)
		for pose in poses:
			pose.draw_pose(surf)
				
		display.blit(surf, (0, 0))
		pygame.display.update()
		pygame.time.delay(15)
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				RUNNING.value = 0
				break
				
	end_time = time.time()
	print ('Tracker2 FPS:', frame_count/(end_time - start_time))
	pygame.quit()

def renderer_allstar(lock, mp_event, face_event):
	global args
	global RUNNING, NFACES, FACE_BOUNDINGBOXES, FACE_FRAMEBUFFER, FACE_RESULTS
	
	pygame.init()
	display = pygame.display.set_mode(appsink_size)
	pygame.display.set_caption('simple_gesture')
	
	frame_count = 0
	start_time = time.time()
	
	target_pose = None
	track_id 	= 0
	tracker 	= SingleTracker()
	stream_ana	= StreamAnalyzer()
	
	face_result = False
	wait_result = False
	
	while RUNNING:
		frame_count += 1
		
		if args.interact:
			mp_event.wait()
			mp_event.clear()
		
		surf, frame = get_surf(lock)
		poses 		= get_pose_shared_data2(lock)
		track_id 	= 0
		
		if len(poses):
			target_pose = None
			if tracker.first_frame:
				target_pose = poses[0]
				tracker.feed(poses, track_id)
			else:			
				match_id = tracker.feed(poses)
				if match_id is not None:
					target_pose = poses[match_id]
					
			if target_pose:
				stream_ana.feed(target_pose)
				target_pose.draw_pose(surf)
				target_pose.draw_boundbox(surf)
				top, bottom = target_pose.get_boundbox()
				
				# print (stream_ana.ana.g_vrotation)
				
				face_boundingbox = stream_ana.ana.get_frontal_face_boundingbox()
				if face_result:
					surf.blit(C_DUY, (target_pose.get_boundbox()[1][0], target_pose.get_boundbox()[0][1]))
					
				if face_boundingbox is not None and not face_event.is_set():
					stream_ana.ana.draw_frontal_face_boundingbox(surf)
					if wait_result:
						if FACE_RESULTS[0]:
							face_result = True
							wait_result = False
						else:
							face_result = False
							# we are tracking the wrong person
							print ('reset tracker')
							tracker.reset()
							wait_result = False
							continue
					
					# face_recognition boundingbox format :)
					FACE_BOUNDINGBOXES[0] = (face_boundingbox[0][1], face_boundingbox[1][0], face_boundingbox[1][1], face_boundingbox[0][0]) 
					NFACES.value = 1
					
					FACE_FRAMEBUFFER[:] = np.ctypeslib.as_ctypes(frame)
					face_event.set()
					wait_result = True
				
				'''
				if ana.g_standing2:
					surf.blit(C_STANDING, (top+bottom)/2)
				elif ana.g_sitting:
					surf.blit(C_SITTING, (top+bottom)/2)
				elif ana.g_lying:
					surf.blit(C_LYING, (top+bottom)/2)
				else:
					surf.blit(C_UNKNOWN, (top+bottom)/2)
				'''
						
				gesture_id = stream_ana.ana.simple_gesture()
				if gesture_id is not None:
					surf.blit(GESTURE_NAMES[gesture_id], target_pose.get_boundbox()[0])
			
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				RUNNING.value = 0
				break
				
		display.blit(surf, (0, 0))
		pygame.display.update()
		pygame.time.delay(15)
	end_time = time.time()
	print ('renderer_allstar:', frame_count/(end_time - start_time)) 
	
def renderer_pose_only(lock, mp_event):
	global args, RUNNING
	
	pygame.init()
	display = pygame.display.set_mode(appsink_size)
	pygame.display.set_caption('pose only')
	
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
		surf, frame = get_surf(lock)
		nposes, kps_coords, kps_scores, pose_scores = get_pose_shared_data(lock)
		
		for i in range(nposes):
			if pose_scores[i] >= C_PSCORE_THRESHOLD:
				draw_pose2(surf, kps_coords[i], kps_scores[i])
				# draw_bound_box(surf, kps_coords[i], kps_scores[i])
					
		display.blit(surf, (0, 0))
		pygame.display.update()
		pygame.time.delay(15)
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				RUNNING.value = 0
				break
				
	end_time = time.time()
	print ('pose only FPS:', frame_count/(end_time - start_time))
	pygame.quit()

def face_recognition(face_event):
	global RUNNING, FACE_FRAMEBUFFER, NFACES, FACE_BOUNDINGBOXES, FACE_RESULTS
	
	import face_recognition, pickle

	#target_image = face_recognition.load_image_file('/home/pi/duy.jpg')
	#known_sig = face_recognition.face_encodings(target_image)[0]
	with open('/home/pi/duy.sig', 'rb') as f:
		known_sig = pickle.loads(f.read())
	print ('sig loaded')
	
	while RUNNING.value:
		face_event.wait()
		face_frame = np.ctypeslib.as_array(FACE_FRAMEBUFFER).copy()
		nfaces = NFACES.value
		face_boundingboxes = []
		for i in range(nfaces):
			face_boundingboxes.append(tuple(FACE_BOUNDINGBOXES[i]))
		
		encodings = face_recognition.face_encodings(face_frame, known_face_locations=face_boundingboxes)
		# print (len(encodings))
		results = face_recognition.compare_faces(encodings, known_sig)
		print (results)
		for i in range(NFACES.value):
			try:
				FACE_RESULTS[i] = results[i]
			except:
				FACE_RESULTS[i] = False
		face_event.clear()
		
def main():
	global RUNNING, PAUSE
	global NPOSES, FRAMEBUFFER, KP_BUFFER, KP_SCORE_BUFFER, POSESCORE_BUFFER 
	global NFACES, FACE_BOUNDINGBOXES, FACE_FRAMEBUFFER, FACE_RESULTS

	FRAMEBUFFER 		= sharedctypes.RawValue(ctypes.c_ubyte*3*appsink_size[0]*appsink_size[1])
	KP_BUFFER			= sharedctypes.RawValue(ctypes.c_int*2*C_NKP*C_MAXPOSE)
	KP_SCORE_BUFFER		= sharedctypes.RawValue(ctypes.c_double*C_NKP*C_MAXPOSE)
	POSESCORE_BUFFER	= sharedctypes.RawValue(ctypes.c_double*C_MAXPOSE)
	
	NPOSES 				= sharedctypes.RawValue(ctypes.c_ushort)
	NFACES				= sharedctypes.RawValue(ctypes.c_ushort)
	FACE_BOUNDINGBOXES	= sharedctypes.RawValue((ctypes.c_ushort*4)*10)
	FACE_RESULTS		= sharedctypes.RawValue(ctypes.c_bool*10)
	FACE_FRAMEBUFFER 	= sharedctypes.RawValue(ctypes.c_ubyte*3*appsink_size[0]*appsink_size[1])
	
	RUNNING				= sharedctypes.RawValue(ctypes.c_ubyte, 1)
	PAUSE				= sharedctypes.RawValue(ctypes.c_ubyte, 0)

	lock 		= mp.Lock()
	mp_event 	= mp.Event()
	face_event 	= mp.Event()
	
	p_face_recognition = mp.Process(target=face_recognition, args=(face_event,))
	p_renderer_tracker 	= mp.Process(target=renderer_tracker, args=(lock, mp_event))
	p_renderer_tracker2 = mp.Process(target=renderer_tracker2, args=(lock, mp_event))
	p_renderer_pose_only = mp.Process(target=renderer_pose_only, args=(lock, mp_event))
	p_renderer_allstar = mp.Process(target=renderer_allstar, args=(lock, mp_event, face_event))
	
	p_renderer_allstar.start()
	p_face_recognition.start()
	
	# p_renderer_tracker2.start()
	# p_renderer_tracker.start()
	# p_renderer_pose_only.start()

	if args.file is not None:
		cap = cv2.VideoCapture(args.file)
	else:
		cap = cv2.VideoCapture(args.cam_id)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, src_size[0])
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, src_size[1])
		'''
		default:
		CAP_PROP_AUTO_EXPOSURE 0(OFF)
		CAP_PROP_EXPOSURE 1000
		'''
		cap.set(cv2.CAP_PROP_EXPOSURE, 1000)
		cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
		print (cap.get(cv2.CAP_PROP_EXPOSURE))
		print (cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
	frame_count = 0
	start_time = time.time()
	pose_height = np.zeros(C_MAXPOSE, dtype=np.uint32)
	while RUNNING:
		if not PAUSE:			
			try:
				frame_count += 1
				# t1 = time.time()
				cap_res, cap_frame 	= cap.read()
				input_img 			= cv2.resize(cap_frame, appsink_size, cv2.INTER_NEAREST)
				input_img 			= cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
				# pil_frame 			= Image.fromarray(input_img)
				
				# input_img = cv2.GaussianBlur(input_img, (5, 5), cv2.BORDER_DEFAULT)
				# opencv ouput frame in (y, x) but use (x, y) point format ¯\_(ツ)_/¯
				# print (input_img.shape)
				nposes, pose_scores, kps, kps_score = pose_engine.DetectPosesInImage(input_img)
				# faces = face_engine.detect_with_image(pil_frame, threshold=0.05, keep_aspect_ratio=False, relative_coord=False, top_k=10)
				# print (faces)
				# t2 = time.time()
				# print ('PoseNet time:', (t2 - t1)*1000)
				lock.acquire()
				FRAMEBUFFER[:] 					= np.ctypeslib.as_ctypes(input_img)
				if nposes:				
					NPOSES.value 				= nposes
					# I converted PoseNet output to (x, y)
					KP_BUFFER[:nposes] 			= np.ctypeslib.as_ctypes(kps.astype(np.int32))
					KP_SCORE_BUFFER[:nposes] 	= np.ctypeslib.as_ctypes(kps_score)
					POSESCORE_BUFFER[:] 		= np.ctypeslib.as_ctypes(pose_scores)						
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
