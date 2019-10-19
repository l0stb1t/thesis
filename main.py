import cv2
import time
import argparse
from operator import itemgetter
from pose_engine import PoseEngine
import numpy as np
from multiprocessing import sharedctypes
import multiprocessing as mp
from math import atan2, degrees, sqrt, pi

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='.tflite model path.', required=False)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
parser.add_argument('--res', help='Resolution', default='640x480', choices=['480x360', '640x480', '1280x720'])
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
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
	if keypoint_score[-1] > 0.3:
		neck = keypoint_coord[-1]
	else:
		neck = None

	if keypoint_score[10] > 0.3:
		r_wrist = keypoint_coord[10]
	else:
		r_wrist = None

	if keypoint_score[9] > 0.3:
		l_wrist = keypoint_coord[9]
	else:
		l_wrist = None

	if keypoint_score[8] > 0.3:
		r_elbow = keypoint_coord[8]
	else:
		r_elbow = None

	if keypoint_score[7] > 0.3:
		l_elbow = keypoint_coord[7]
	else:
		l_elbow = None

	if keypoint_score[6] > 0.3:
		r_shoulder = keypoint_coord[6]
	else:
		r_shoulder = None

	if keypoint_score[5] > 0.3:
		l_shoulder = keypoint_coord[5]
	else:
		l_shoulder = None

	if keypoint_score[4] > 0.3:
		r_ear = keypoint_coord[4]
	else:
		r_ear = None

	if keypoint_score[3] > 0.3:
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


def renderer():
	global sharedmem, child_cnx

	while (1):
		if child_cnx.poll():
			if child_cnx.recv() == 's':
				break

	frame_count = 0
	start_time = time.time()
	while (1):
		frame_count += 1
		frame = np.ctypeslib.as_array(sharedmem).copy()
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		time.sleep(0.01)
	end_time = time.time()
	print (frame_count/(end_time - start_time))
	child_cnx.send('q')

def draw_pose(img, pose):
	cv2_keypoints = []
	for keypoint_name in pose.keypoints:
		keypoint = pose.keypoints[keypoint_name]
		if (keypoint.score > 0.5):
			cv2_keypoints.append(cv2.KeyPoint(keypoint.yx[1], keypoint.yx[0], 20.0*keypoint.score))
	out_img = cv2.drawKeypoints(img, cv2_keypoints, outImage=np.array([]), color=(255, 0, 0))
	return out_img

def main():
	global sharedmem, parent_cnx, child_cnx

	frame = np.zeros((appsink_size[1], appsink_size[0], 3) , dtype=np.uint8)
	ctype_frame = np.ctypeslib.as_ctypes(frame)
	sharedmem = sharedctypes.RawArray(ctype_frame._type_, (ctype_frame))

	#create child process
	parent_cnx, child_cnx = mp.Pipe()

	p_renderer = mp.Process(target=renderer)
	p_renderer.start()
	#signal child process to start
	parent_cnx.send('s')

	if args.file is not None:
		cap = cv2.VideoCapture(args.file)
	else:
		cap = cv2.VideoCapture(args.cam_id)

	cap.set(3, appsink_size[0])
	cap.set(4, appsink_size[1])

	frame_count = 0
	start_time = time.time()
	while True:
		frame_count += 1

		prepare_input_time = time.monotonic()
		cap_res, cap_frame = cap.read()
		input_img = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
		prepare_input_time = time.monotonic() - prepare_input_time

		decode_pose_time = time.monotonic()
		outputs, inference_time = engine.DetectPosesInImage(input_img)
		if len(outputs) == 0:
			sharedmem[:] = np.ctypeslib.as_ctypes(cap_frame.copy())
			continue
		max_pose = max(outputs, key=lambda i: i.score)
		out_img = draw_pose(cap_frame, max_pose)
		decode_pose_time = time.monotonic() - decode_pose_time
		
		# write frame to memory
		render_time = time.monotonic()
		sharedmem[:] = np.ctypeslib.as_ctypes(out_img.copy())
		if parent_cnx.poll():
			if parent_cnx.recv() == 'q':
				break
		render_time = time.monotonic() - render_time

		print (prepare_input_time, decode_pose_time, render_time)

	end_time = time.time()
	print (frame_count/(end_time - start_time))

if __name__ == "__main__":
	main()