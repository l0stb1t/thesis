import sys
sys.path.insert(0, 'tello-posenet')

import re
import os
import av
import cv2
import time
import random
import ctypes
import logging
import tellopy
import argparse
import datetime 
import traceback
import numpy as np
import multiprocessing as mp

from util import *
from constants import *
from drone_constants import *

from PN import PoseNet
from simple_pid import PID
from pose_engine import PoseEngine
from math import pi, atan2, degrees, sqrt
from multiprocessing import Process, sharedctypes

logging.basicConfig()
LOGGER = logging.getLogger("TellPoseNet2")
LOGGER.setLevel(logging.CRITICAL)
av.logging.set_level(av.logging.PANIC)

def get_surf(lock):
    global FRAMEBUFFER
    
    #lock.acquire()
    frame = np.ctypeslib.as_array(FRAMEBUFFER).copy()
    #lock.release()
    ''' Pygame use (x, y) frame format so have to convert it but return the original frame for later use with opencv '''
    surf = pygame.surfarray.make_surface(np.rot90(frame)[::-1])
    return surf, frame

def recorder():
	global args
	global FRAMEBUFFER, RUNNING
	
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	try:
		writer = cv2.VideoWriter(args.record, fourcc, 20, (480, 360))
		while RUNNING:
			frame = np.ctypeslib.as_array(FRAMEBUFFER).copy()
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			writer.write(frame)
	except:
		traceback.print_exc()

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

def pilot(sockfd, lock, face_event):
	global FRAMEBUFFER, RUNNING, LOGGER
	
	use_autopilot = False
	display = init_pygame_window('tello')
	drone 	= tellopy.Tello(sockfd=sockfd, no_video_thread=True)
	drone.connect()
	
	posenet = PoseNet(model='models/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite')
	
	controller = TelloController(drone)
	autopilot = AutoPilot(controller, posenet=posenet)
	fps = FPS()
	
	
	while RUNNING:
		surf, frame = get_surf(lock)
		fps.update()
		fps.display(surf)
		
		poses = posenet.eval(frame)

		try:
			if use_autopilot:
				target_pose = autopilot.feed(frame, poses)
				for pose in poses:
					pose.draw_pose(surf)
				#if (target_pose):
				#	target_pose.draw_pose(surf)
				
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					RUNNING.value = 0
					break
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_F12:
						LOGGER.info('AutoPilot activated')
						use_autopilot = True
					else:
						if use_autopilot == True:
							use_autopilot = False
							autopilot.reset()
						controller.handle_keydown(event.key)
				elif event.type == pygame.KEYUP:
					controller.handle_keyup(event.key)
		except:
			''' fail safe in case of programming error'''
			RUNNING.value = 0
			controller.failsafe()
			print (traceback.print_exc())
			break

		display.blit(surf, (0, 0))
		pygame.display.update()
		pygame.time.delay(5)
	pygame.quit()

def init_drone():
	''' we don't recv data here, only recv video '''
	drone = tellopy.Tello(no_recv_thread=True) 
	drone.connect()
	drone.set_video_encoder_rate(2)
	drone.start_video()
	drone.set_loglevel(drone.LOG_ERROR)
	
	return drone

def main():
	global FRAMEBUFFER, RUNNING
	global NFACES, FACE_BOUNDINGBOXES, FACE_RESULTS, FACE_FRAMEBUFFER
	
	RUNNING		= sharedctypes.RawValue(ctypes.c_ubyte, 1)
	FRAMEBUFFER = sharedctypes.RawValue(ctypes.c_ubyte*3*480*360)
	
	NFACES				= sharedctypes.RawValue(ctypes.c_ushort)
	FACE_BOUNDINGBOXES	= sharedctypes.RawValue((ctypes.c_ushort*4)*10)
	FACE_RESULTS		= sharedctypes.RawValue(ctypes.c_bool*10)
	FACE_FRAMEBUFFER 	= sharedctypes.RawValue(ctypes.c_ubyte*3*480*360)
	
	drone = init_drone()
	
	if args.record:
		p_recorder = Process(target=recorder)
		p_recorder.start()
	
	face_event 	= mp.Event()
	lock 		= mp.Lock()
	p_pilot 	= Process(target=pilot, args=(drone.sock.fileno(), lock, face_event))
	p_pilot.start()
	
	if args.face:
		p_face_recognition = mp.Process(target=face_recognition, args=(face_event,))
		p_face_recognition.start()
	
	container = av.open(drone.get_video_stream())
	frame_skip = 350
	fps = FPS()
	for frame in container.decode(video=0):
		fps.update()
		if not RUNNING.value:
			break
			
		if 0 < frame_skip:
			frame_skip = frame_skip - 1
			continue
		start_time = time.time()
		if frame.time_base < 1.0/60:
			time_base = 1.0/60
		else:
			time_base = frame.time_base

		# Convert frame to cv2 image
		frame = frame.to_ndarray(width=480, height=360, format='bgr24')
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		FRAMEBUFFER[:] = np.ctypeslib.as_ctypes(frame)
		cur = time.time()
		prev = cur
		frame_skip = int((time.time() - start_time)/time_base)
	print ('VideoReceiver FPS:', fps.get())

class TelloController():
	def __init__(self, drone, media_directory='media', log_level=logging.ERROR):
		self.drone = drone
		
		try:
			os.mkdir(media_directory) # :)
		except:
			pass
		self.media_directory = media_directory
		
		# for debugging
		self.logger = logging.getLogger("TelloController")
		self.logger.setLevel(log_level)
		
		# Flight data
		self.is_flying 		 = False
		self.battery 		 = None
		self.fly_mode 		 = None
		self.throw_fly_timer = 0
		
		self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, 		self.flight_data_handler)
		self.drone.subscribe(self.drone.EVENT_LOG_DATA, 		self.log_data_handler)
		self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED, 	self.file_received_handler)
		
		self.init_state()
		self.init_keymapping()
	
	''' Keyboard related methods '''
	def init_keymapping(self):
		''' Define keys and add listener '''
		
		self.keydown_mapping = {
			pygame.K_w: lambda: self.drone.forward(C_PITCH_SPEED),
			pygame.K_s: lambda: self.drone.forward(-C_PITCH_SPEED),
			pygame.K_a: lambda: self.drone.right(-C_ROLL_SPEED),
			pygame.K_d: lambda: self.drone.right(C_ROLL_SPEED),

			pygame.K_F1: lambda: self.drone.takeoff(),
			pygame.K_F2: lambda: self.drone.land(),
			pygame.K_F3: lambda: self.drone.flip_forward(),
			pygame.K_F4: lambda: self.drone.flip_back(),
			pygame.K_F5: lambda: self.drone.flip_left(),
			pygame.K_F6: lambda: self.drone.flip_right(),
			
			pygame.K_LEFT: 	lambda: self.drone.clockwise(-1.5*C_YAW_SPEED),
			pygame.K_RIGHT: lambda: self.drone.clockwise(1.5*C_YAW_SPEED),
			pygame.K_UP: 	lambda: self.drone.up(C_THROTTLE_SPEED),
			pygame.K_DOWN: 	lambda: self.drone.up(-C_THROTTLE_SPEED),
			
			pygame.K_RETURN: lambda: self.take_picture(),
			pygame.K_ESCAPE: self.failsafe,
			
			pygame.K_p: lambda: self.palm_land(),
			#pygame.K_t: lambda: self.toggle_tracking(),
			#pygame.K_o: lambda: self.toggle_posenet(),
			#pygame.K_c: lambda: self.clockwise_degrees(360),
			pygame.K_0: lambda: self.drone.set_video_encoder_rate(0),
			pygame.K_1: lambda: self.drone.set_video_encoder_rate(1),
			pygame.K_2: lambda: self.drone.set_video_encoder_rate(2),
			pygame.K_3: lambda: self.drone.set_video_encoder_rate(3),
			pygame.K_4: lambda: self.drone.set_video_encoder_rate(4),
			pygame.K_5: lambda: self.drone.set_video_encoder_rate(5),

			pygame.K_7: lambda: self.set_exposure(-1),	
			pygame.K_8: lambda: self.set_exposure(0),
			pygame.K_9: lambda: self.set_exposure(1),
			# 'z': lambda: self.delayed_takeoff(),
		}

		self.keyup_mapping = {
			pygame.K_w: lambda: self.drone.forward(0),
			pygame.K_s: lambda: self.drone.forward(0),
			pygame.K_a: lambda: self.drone.right(0),
			pygame.K_d: lambda: self.drone.right(0),
			
			pygame.K_LEFT: 	lambda: self.drone.clockwise(0),
			pygame.K_RIGHT: lambda: self.drone.clockwise(0),
			pygame.K_UP: 	lambda: self.drone.up(0),
			pygame.K_DOWN: 	lambda: self.drone.up(0)
		}

	def handle_keyup(self, key):
		try:
			self.keyup_mapping[key]()
		except KeyError:
			pass
		
	def handle_keydown(self, key):
		try:
			self.keydown_mapping[key]()
		except KeyError:
			pass
	
	''' Drone related methods '''
	def set_video_encoder_rate(self, rate):
		self.drone.set_video_encoder_rate(rate)
		self.video_encoder_rate = rate
		
	def init_state(self):
		''' Reset global variables before a fly '''
		self.logger.info("reset")
		self.ref_pos_x = None
		self.ref_pos_y = None
		self.ref_pos_z = None
		self.pos_x = None
		self.pos_y = None
		self.pos_z = None
		self.timestamp_take_picture = None
		self.throw_ongoing = False
		self.scheduled_takeoff = None
		
	def failsafe(self):
		self.logger.critical('failsafe')
		self.drone.up(0)
		self.drone.right(0)
		self.drone.forward(0)
		self.drone.clockwise(0)
		
	def stay(self):
		self.logger.debug('stay')
		self.drone.up(0)
		self.drone.right(0)
		self.drone.forward(0)
		self.drone.clockwise(0)
		
	def throttle(self, speed):
		self.logger.debug('throttle %d' % speed)
		self.drone.up(speed)

	def roll(self, speed):
		self.logger.debug('roll %d' % speed)
		self.drone.right(speed)
		
	def pitch(self, speed):
		self.logger.debug('pitch %d' % speed)
		self.drone.forward(speed)
		
	def yaw(self, speed):
		self.logger.debug('yaw %d' % speed)
		self.drone.clockwise(speed)

	def spin(self, speed):
		self.logger.debug('spin %d' % speed)
		self.drone.up(0)
		self.drone.right(0)
		self.drone.forward(0)
		self.drone.clockwise(30)

	def take_picture(self):
		''' Tell drone to take picture, image sent to file handler '''
		self.logger.info('taking picture')
		self.drone.take_picture()

	def set_exposure(self, expo):
		''' 
		0 set exposure to 0
		1 increase exposure
		-1 decrease exposure
		'''
		if expo == 0:
			self.exposure = 0
		elif expo == 1:
			self.exposure = min(9, self.exposure+1)
		elif expo == -1:
			self.exposure = max(-9, self.exposure-1)
		self.drone.set_exposure(self.exposure)
		self.logger.info('exposure %d' % self.exposure)

	def land(self):
		self.logger.info('landing')
		self.drone.land()

	def palm_land(self):
		self.logger.info('palm landing')
		self.drone.palm_land()

	def throw_and_go(self):
		self.logger.info('throw and go')
		self.drone.throw_and_go()	  

	def flight_data_handler(self, event, sender, data):
		''' Listener to flight data from the drone. '''
		self.battery = data.battery_percentage
		if (self.battery < 15):
			self.logger.critical('Batter low')
		
		if self.is_flying != data.em_sky:
			self.is_flying = data.em_sky
			self.logger.debug('flying %d' % self.is_flying)
			if not self.is_flying:
				''' restart '''
				self.init_state()
				
	def log_data_handler(self, event, sender, data):
		''' Listener to log data from the drone. '''
		pos_x = -data.mvo.pos_x
		pos_y = -data.mvo.pos_y
		pos_z = -data.mvo.pos_z
		if abs(pos_x)+abs(pos_y)+abs(pos_z) > 0.07:
			''' First time we have meaningful values, we store them as reference '''
			if self.ref_pos_x is None:
				self.ref_pos_x = pos_x
				self.ref_pos_y = pos_y
				self.ref_pos_z = pos_z
			else:
				self.pos_x = pos_x-self.ref_pos_x
				self.pos_y = pos_y-self.ref_pos_y
				self.pos_z = pos_z-self.ref_pos_z
		
		qx = data.imu.q1
		qy = data.imu.q2
		qz = data.imu.q3
		qw = data.imu.q0
		
		# print (self.pos_z)
		# self.yaw = quat_to_yaw_deg(qx,qy,qz,qw)
		
	def file_received_handler(self, event, sender, data):
		''' Create a file in local directory to receive image from the drone '''
		date_fmt = '%Y-%m-%d_%H%M%S'
		path = f'{self.media_directory}/tello-{datetime.datetime.now().strftime(date_fmt)}.jpg' 
		with open(path, 'wb') as out_file:
			out_file.write(data)
		self.logger.info('Photo saved to %s' % path)

'''
Fly the drone automatically 
Input: Frame
Output: speed vector
You can choose to use face recognition or not
'''
class AutoPilot():
	def __init__(self, controller, posenet=None, use_face_recognition=True, log_level=logging.DEBUG):			
		self.controller = controller
		self.tracker 	= SingleTracker()
		self.stream_ana = StreamAnalyzer()
		
		self.land = False
		
		''' we can use a PoseNet model from outside or we can create our own '''
		if posenet:
			self.posenet = posenet
		else:
			self.posenet = PoseNet()
		
		self.use_face_recognition = use_face_recognition
		if self.use_face_recognition:
			self.face_recognition = FaceRecognition()
		self.found_right_person = False
		self.false_count = 0
		
		self.state = C_STATE_SEARCHING
		self.logger = logging.getLogger('AutoPilot')
		self.logger.setLevel(log_level)
		
		self.reset()
		
		self.gesture_counter = 0
		self.no_pose_counter = 0
		
		self.face_lock = False
		self.timer = None
		self.spin_timer = None
		
	def choose_random_target_pose(self, poses):
		target_pose = None
		''' choose a random pose to track '''
		idx = random.randint(0, len(poses)-1)
		target_pose = poses[idx]
		self.tracker.feed(poses, idx)
		return target_pose
		
	def get_next_target_pose(self, poses):
		match_idx = self.tracker.feed(poses)
		if match_idx is not None:
			target_pose = poses[match_idx]
		else:
			target_pose = None
		return target_pose
	
	def get_surf(self):
		if self.surf is None:
			self.surf = pygame.Surface((C_WIDTH, C_HEIGHT))
		return self.surf
	
	def pid_control(self, target_pose):
		kp_order = (C_NOSE, C_NECK, C_MIDHIP)
		for kp_id in kp_order:
			kp = target_pose.has_kp(kp_id)
			if kp:
				target_kp = kp
				if kp_id == C_NOSE:
					# self.logger.debug('Found C_NOSE')
					center_x = int(C_WIDTH*0.5)
					center_y = int(C_HEIGHT*0.35)
				elif kp_id == C_NECK:
					# self.logger.debug('Found C_NECK')
					center_x = int(C_WIDTH*0.5)
					center_y = int(C_HEIGHT*0.5)
				elif kp_id == C_MIDHIP:
					# self.logger.debug('Found C_MIDHIP')
					center_x = int(C_WIDTH*0.5)
					center_y = int(C_HEIGHT*0.75)
				break
				
		x_offset = center_x - target_kp.xy[0]
		y_offset = target_kp.xy[1] - center_y
		
		d_offset = None
		if self.stream_ana.ana.g_eyes_distance and self.stream_ana.ana.get_frontal_face_boundingbox() is not None:
			if self.keep_distance is not None:
				if self.keep_distance < 15:
					self.keep_distance = 15
				elif self.keep_distance > 30:
					self.keep_distance = 30
				d_offset = self.keep_distance - self.stream_ana.ana.g_eyes_distance
			else:
				self.keep_distance = self.stream_ana.ana.g_eyes_distance				
	
		rotation = None
		if self.stream_ana.ana.g_rotation is not None:
			rotation = self.stream_ana.ana.g_rotation
		
		yaw_speed = self.pid_yaw(x_offset)
		# self.logger.debug('yaw speed: %d' % yaw_speed)
		self.controller.yaw(yaw_speed)
		
		if self.face_lock and rotation is not None:
			self.controller.roll(self.pid_roll(rotation))
		else:
			self.controller.roll(0)
			
		self.controller.throttle(self.pid_throttle(y_offset))
		
		if d_offset is not None:
			pitch_speed = -self.pid_pitch(d_offset)
			# self.logger.debug('pitch speed: %d' % pitch_speed)
			self.controller.pitch(pitch_speed)
		else: 
			self.controller.pitch(0)
		
	
	''' search for the right person to track '''
	def search(self, frame, poses):
		if self.spin_timer is not None:
			self.controller.spin(30)
			if self.spin_timer == 0:
				self.spin_timer = None
			else:
				self.spin_timer -= 1
			return
			
		if len(poses):
			self.no_pose_counter = 0
			''' there are people in frame '''
			target_pose = None
			
			if self.tracker.first_frame:
				target_pose = self.choose_random_target_pose(poses)
			else:
				''' we are already tracking someone, continue to follow them for face recognition '''
				target_pose = self.get_next_target_pose(poses)
			
			if target_pose is None:
				self.logger.debug('loose track spinning')
				self.tracker.reset()
				self.controller.spin(30)
				return None
			else:					
				''' we currently having a target pose '''
				self.stream_ana.feed(target_pose)
				
				if self.stream_ana.ana.g_eyes_distance is not None:
					if self.stream_ana.ana.g_eyes_distance<20:
						self.logger.debug('pose too far, keep spinning')
						self.spin_timer = 60
						return

				if self.face_recognition.waiting_result:
					result = self.face_recognition.has_result()
					if result is not None:
						if result[0] is False:
							self.false_count += 1
							self.logger.debug('false_count: %d' % self.false_count) 
							if self.false_count == 10:
								self.false_cout = 0
								self.logger.debug('wrong person')
								self.found_right_person = False
								self.reset()
								self.spin_timer = 60
						else:
							self.logger.debug('right person')
							self.logger.debug('C_STATE_FOLLOWING')
							self.found_right_person = True
							self.face_lock = False
							self.state = C_STATE_FOLLOWING
							return target_pose
				else:
					face_boundingbox = self.stream_ana.ana.get_frontal_face_boundingbox()
					if face_boundingbox is not None:
						top, bottom = face_boundingbox
						brightness = np.mean(frame[top[1]: bottom[1], top[0]: bottom[0]])
						if brightness >= 80 and brightness <= 130:
							self.face_recognition.feed([face_boundingbox], frame)
						
				''' try to move as little as possible '''
				self.keep_distance = None
				self.face_lock = True
				self.pid_control(target_pose)
				return target_pose
					
		else:
			''' there are nobody in the current frame '''
			self.no_pose_counter += 1
			if self.no_pose_counter > 500:
				self.controller.failsafe()
				
			elif self.no_pose_counter > 50:
				self.logger.debug('no pose spinning')
				self.tracker.reset()
				self.controller.spin(30)
				
	def handle_gesture_cmd(self):
		if self.timer is not None:
			if self.timer == 0:
				self.timer = None
				self.controller.take_picture()
			else:
				self.timer -= 1
				
		if self.gesture_counter>0:
			self.gesture_counter -= 1
		if self.gesture_counter==0:
			self.gesture_counter = C_GESTURE_COUNTER
			gesture = self.stream_ana.simple_gesture()
			if gesture == C_LEFT_ARM_UP_CLOSED:
				self.logger.info('C_LEFT_ARM_UP_CLOSED')
				if self.keep_distance is not None:
					self.keep_distance -= 5
					self.logger.info('keep_distance %d' % self.keep_distance)
					
			elif gesture == C_LEFT_ARM_UP_OPEN:
				self.logger.info('C_LEFT_ARM_UP_OPEN')
				if self.keep_distance is not None:
					self.keep_distance += 5
					self.logger.info('keep_distance %d' % self.keep_distance)
			
			elif gesture == C_HANDS_ON_NECK:
				self.logger.info('C_HANDS_ON_NECK')
				self.face_lock ^= True
				self.logger.info('face_lock: ' + str(self.face_lock))
			
			elif gesture == C_CLOSE_HANDS_UP:
				self.logger.info('C_CLOSE_HANDS_UP')
				self.state = C_STATE_STOP
				
			elif gesture == C_RIGHT_ARM_UP_CLOSED:
				self.logger.info('C_RIGHT_ARM_UP_CLOSED')
				self.timer = 60
			
	def follow(self, frame, poses):
		if len(poses):
			self.no_pose_counter = 0
			''' there are people in frame '''
			target_pose = None
			
			if self.tracker.first_frame:
				target_pose = self.choose_random_target_pose(poses)
			else:
				''' we are already tracking someone, continue to follow them for face recognition '''
				target_pose = self.get_next_target_pose(poses)
			
			if target_pose is None:
				self.logger.debug('loose track C_STATE_SEARCHING')
				self.reset()
				return None
			else:
				''' we currently having a target pose '''
				self.stream_ana.feed(target_pose)
				if self.face_recognition.waiting_result:
					result = self.face_recognition.has_result()
					if result is not None:
						if result[0] is False:
							self.false_count += 1
							self.logger.debug('false_count: %d' % self.false_count) 
							if (self.false_count == 10):
								self.false_count = 0
								self.logger.debug('wrong person C_STATE_SEARCHING')
								self.reset()
								return
						else:
							self.false_count = 0
							self.logger.debug('right person')
				else:
					face_boundingbox = self.stream_ana.ana.get_frontal_face_boundingbox()
					if face_boundingbox is not None and self.stream_ana.ana.g_eyes_distance>=20:
						top, bottom = face_boundingbox
						brightness = np.mean(frame[top[1]: bottom[1], top[0]: bottom[0]])
						if brightness >= 80 and brightness <= 130:
							self.face_recognition.feed([face_boundingbox], frame)
				self.handle_gesture_cmd()				
				self.pid_control(target_pose)
				return target_pose
					
		else:
			''' there are nobody in the current frame '''
			self.no_pose_counter += 1
			if self.no_pose_counter > 1000:
				self.controller.failsafe()
				
			elif self.no_pose_counter > 50:
				self.logger.debug('no pose C_STATE_SEARCHING')
				self.reset()
				return
										
	def feed(self, frame, poses):
		self.surf = None
		if self.state == C_STATE_SEARCHING:
			return self.search(frame, poses)
		elif self.state == C_STATE_FOLLOWING:
			return self.follow(frame, poses)
		elif self.state == C_STATE_STOP:
			if not self.land:
				self.logger.debug('landing')
				self.land = True
				self.controller.stay()
				self.controller.land()
			return

			'''
			pygame.draw.circle(surf, C_BLUE, (center_x, center_y), 3)
			pygame.draw.line(surf, C_GREEN, (center_x, center_y), target_kp.xy, 3)
			'''
			
	def reset(self):					
		self.logger.info('reset')
		self.pid_roll 		= PID(C_ROLL_KP, 		C_ROLL_KI, 		C_ROLL_KD, 		setpoint=0, output_limits=(-50, 50))
		self.pid_yaw 		= PID(C_YAW_KP, 		C_YAW_KI, 		C_YAW_KD, 		setpoint=0, output_limits=(-100, 100))
		self.pid_pitch		= PID(C_PITCH_KP, 		C_PITCH_KI, 	C_PITCH_KD, 	setpoint=0, output_limits=(-100, 100))
		self.pid_throttle 	= PID(C_THROTTLE_KP, 	C_THROTTLE_KI, 	C_THROTTLE_KD, 	setpoint=0, output_limits=(-100, 100))
		self.tracker.reset()
		self.keep_distance = None
		self.found_right_person = False
		self.false_count = 0
		self.state = C_STATE_SEARCHING
		self.controller.stay()


if __name__ == '__main__':
	global args
	
	ap=argparse.ArgumentParser()
	ap.add_argument('-face', action='store_true')
	ap.add_argument('-record', type=str, help='video output path')
	args=ap.parse_args()
	main()
