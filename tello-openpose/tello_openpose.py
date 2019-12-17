"""
tello_openpose.py : Use the Tello drone as an "selfie air stick"
Relies on tellopy (for interaction with the Tello drone) and Openpose (for body detection and pose recognition)

I started from: https://github.com/Ubotica/telloCV/blob/master/telloCV.py 

"""
import sys
sys.path.insert(0, '/home/pi/thesis')
from constants import *

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

import numpy as np
import av, cv2, tellopy, ctypes

from math import pi, atan2, degrees, sqrt
import os, time, datetime, re, logging, argparse
import multiprocessing as mp
from multiprocessing import Process, sharedctypes


from simple_pid import PID
from pynput import keyboard
from pose_engine import PoseEngine

from FPS import FPS
from CameraMorse import CameraMorse, RollingGraph
from PN import *

log = logging.getLogger("TellOpenpose")
log.setLevel(logging.CRITICAL)
av.logging.set_level(av.logging.PANIC)

#def distance(A, B):
#	return int(sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2))

def distance (A, B):
	return np.linalg.norm(A-B)

def angle (A, B, C):
	''' Calculate the angle between segment(A,p2) and segment (p2,p3) '''
	if A is None or B is None or C is None:
		return None
	return degrees(atan2(C[1]-B[1],C[0]-B[0]) - atan2(A[1]-B[1],A[0]-B[0]))%360

def vertical_angle(A, B):
	''' Calculate the angle between segment(A,B) and vertical axe '''
	return degrees(atan2(B[1]-A[1],B[0]-A[0]) - pi/2)

def quat_to_yaw_deg(qx,qy,qz,qw):
	''' Calculate yaw from quaternion '''
	degree = pi/180
	sqy = qy*qy
	sqz = qz*qz
	siny = 2 * (qw*qz+qx*qy)
	cosy = 1 - 2*(qy*qy+qz*qz)
	yaw = int(atan2(siny,cosy)/degree)
	return yaw

def openpose_worker():
	''' In 2 processes mode, this is the init and main loop of the child '''
	print("Worker process",os.getpid())
	tello.drone.start_recv_thread()
	tello.init_controls()
	tello.op = OP(number_people_max=1, min_size=25, debug=tello.debug)

	while True:
		tello.fps.update()
		frame 		= np.ctypeslib.as_array(tello.shared_array).copy()
		frame.shape	= tello.frame_shape
		frame 		= tello.process_frame(frame)

		cv2.imshow("Processed", frame)
		cv2.waitKey(1)
		
def worker(sockfd, lock):
	global FRAMEBUFFER
		
	drone = tellopy.Tello(sockfd=sockfd, no_video_thread=True)
	drone.connect()
	tello = TelloController(drone, write_log_data=False, log_level=logging.CRITICAL)
	while True:
		lock.acquire()
		frame = np.ctypeslib.as_array(FRAMEBUFFER).copy()
		lock.release()
		
		tello.fps.update()
		frame = tello.process_frame(frame)
		cv2.imshow("Processed", frame)
		cv2.waitKey(1)
	
def main(log_level=None):
	global FRAMEBUFFER
	FRAMEBUFFER = sharedctypes.RawValue(ctypes.c_ubyte*3*480*360)
		
	drone = tellopy.Tello(no_recv_thread=True)
	drone.connect()
	drone.set_video_encoder_rate(2)
	drone.start_video()
	drone.set_loglevel(drone.LOG_ERROR)
	
	lock 		= mp.Lock()
	p_worker 	= Process(target=worker, args=(drone.sock.fileno(), lock))
	p_worker.start()
	
	container 	= av.open(drone.get_video_stream())
	prev = time.time()
	frame_skip = 300
	for frame in container.decode(video=0):
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
		FRAMEBUFFER[:] = np.ctypeslib.as_ctypes(frame)
		cur = time.time()
		# print (cur-prev)
		prev = cur
		frame_skip = int((time.time() - start_time)/time_base)
	
class TelloController(object):
	'''TelloController builds keyboard controls on top of TelloPy as well as generating images from the video stream and enabling opencv support'''

	def __init__(self, drone, kbd_layout='QWERTY',  write_log_data=False, media_directory='media', log_level=logging.CRITICAL):
		if log_level is None:
			log_level = logging.CRITICAL
		self.write_log_data = write_log_data
		self.drone = drone
		self.log_level 	= log_level
		self.debug 		= False
		self.kbd_layout = kbd_layout
		
		# Flight data
		self.is_flying 		 = False
		self.battery 		 = None
		self.fly_mode 		 = None
		self.throw_fly_timer = 0

		self.tracking_after_takeoff = False
		self.record = False
		self.keydown = False
		self.date_fmt = '%Y-%m-%d_%H%M%S'
	
		self.axis_command 				= np.zeros(4, dtype=np.object)
		self.axis_command[C_YAW] 		= self.drone.clockwise
		self.axis_command[C_ROLL] 		= self.drone.right
		self.axis_command[C_PITCH] 		= self.drone.forward
		self.axis_command[C_THROTTLE] 	= self.drone.up
				
		self.axis_speed 		= np.zeros(4, dtype=np.int32)
		self.cmd_axis_speed 	= np.zeros(4, dtype=np.int32)	 
		self.prev_axis_speed 	= self.axis_speed.copy()
		self.def_speed 			= np.zeros(4, dtype=np.int32)
		
		self.def_speed[C_YAW] 		= 50
		self.def_speed[C_ROLL] 		= 35
		self.def_speed[C_PITCH] 	= 35
		self.def_speed[C_THROTTLE] 	= 80
		
		self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, 		self.flight_data_handler)
		self.drone.subscribe(self.drone.EVENT_LOG_DATA, 		self.log_data_handler)
		self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED, 	self.handle_flight_received)
		
		self.init_controls()
		
		# Setup PoseNet
		self.op = PN()
		self.use_posenet = True
			
		self.morse = CameraMorse(display=False)
		self.morse.define_command("---", self.delayed_takeoff)
		self.morse.define_command("...", self.throw_and_go, {'tracking':True})
		self.is_pressed = False
	   
		self.fps 		= FPS()
		self.exposure 	= 0
		
		self.reset()
		self.media_directory = media_directory
		if not os.path.isdir(self.media_directory):
			os.makedirs(self.media_directory)

		if self.write_log_data:
			path = 'tello-%s.csv' % datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
			self.log_file = open(path, 'w')
			self.write_header = True

		if log_level is not None:
			if log_level == "info":
				log_level = logging.INFO
			elif log_level == "debug":
				log_level = logging.DEBUG
			log.setLevel(log_level)
			ch = logging.StreamHandler(sys.stdout)
			ch.setLevel(log_level)
			ch.setFormatter(logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
							datefmt="%H:%M:%S"))
			log.addHandler(ch)
		
	def set_video_encoder_rate(self, rate):
		self.drone.set_video_encoder_rate(rate)
		self.video_encoder_rate = rate

	def reset(self):
		''' Reset global variables before a fly '''
		log.debug("RESET")
		self.ref_pos_x = -1
		self.ref_pos_y = -1
		self.ref_pos_z = -1
		self.pos_x = -1
		self.pos_y = -1
		self.pos_z = -1
		self.yaw = 0
		self.tracking = False
		self.keep_distance = None
		self.palm_landing = False
		self.palm_landing_approach = False
		self.yaw_to_consume = 0
		self.timestamp_keep_distance = time.time()
		self.wait_before_tracking = None
		self.timestamp_take_picture = None
		self.throw_ongoing = False
		self.scheduled_takeoff = None
		# When in trackin mode, but no body is detected in current frame,
		# we make the drone rotate in the hope to find some body
		# The rotation is done in the same direction as the last rotation done
		self.body_in_prev_frame = False
		self.timestamp_no_body = time.time()
		self.last_rotation_is_cw = True

	def on_press(self, keyname):
		''' Handler for keyboard listener '''
		if self.keydown:
			return
		try:
			self.keydown = True
			keyname = str(keyname).strip('\'')
			log.info('KEY PRESS ' + keyname)
			if keyname == 'Key.esc':
				self.toggle_tracking(False)
				# self.tracking = False
				self.drone.land()
				self.drone.quit()
				cv2.destroyAllWindows() 
				os._exit(0)
			if keyname in self.controls_keypress:
				self.controls_keypress[keyname]()
		except AttributeError:
			log.debug(f'special key {keyname0} pressed')

	def on_release(self, keyname):
		''' Reset on key up from keyboard listener '''
		self.keydown = False
		keyname = str(keyname).strip('\'')
		log.info('KEY RELEASE ' + keyname)
		if keyname in self.controls_keyrelease:
			key_handler = self.controls_keyrelease[keyname]()
	
	def set_speed(self, axis, speed):
		log.debug(f"set speed {axis} {speed}")
		self.cmd_axis_speed[axis] = speed

	def init_controls(self):
		''' Define keys and add listener '''
		
		controls_keypress_QWERTY = {
			'w': lambda: self.set_speed(C_PITCH, self.def_speed[C_PITCH]),
			's': lambda: self.set_speed(C_PITCH, -self.def_speed[C_PITCH]),
			'a': lambda: self.set_speed(C_ROLL, -self.def_speed[C_ROLL]),
			'd': lambda: self.set_speed(C_ROLL, self.def_speed[C_ROLL]),
			'q': lambda: self.set_speed(C_YAW, -self.def_speed[C_YAW]),
			'e': lambda: self.set_speed(C_YAW, self.def_speed[C_YAW]),
			'i': lambda: self.drone.flip_forward(),
			'k': lambda: self.drone.flip_back(),
			'j': lambda: self.drone.flip_left(),
			'l': lambda: self.drone.flip_right(),
			'Key.left': lambda: self.set_speed(C_YAW, -1.5*self.def_speed[C_YAW]),
			'Key.right': lambda: self.set_speed(C_YAW, 1.5*self.def_speed[C_YAW]),
			'Key.up': lambda: self.set_speed(C_THROTTLE, self.def_speed[C_THROTTLE]),
			'Key.down': lambda: self.set_speed(C_THROTTLE, -self.def_speed[C_THROTTLE]),
			'Key.tab': lambda: self.drone.takeoff(),
			'Key.backspace': lambda: self.drone.land(),
			'p': lambda: self.palm_land(),
			't': lambda: self.toggle_tracking(),
			'o': lambda: self.toggle_openpose(),
			'Key.enter': lambda: self.take_picture(),
			'c': lambda: self.clockwise_degrees(360),
			'0': lambda: self.drone.set_video_encoder_rate(0),
			'1': lambda: self.drone.set_video_encoder_rate(1),
			'2': lambda: self.drone.set_video_encoder_rate(2),
			'3': lambda: self.drone.set_video_encoder_rate(3),
			'4': lambda: self.drone.set_video_encoder_rate(4),
			'5': lambda: self.drone.set_video_encoder_rate(5),

			'7': lambda: self.set_exposure(-1),	
			'8': lambda: self.set_exposure(0),
			'9': lambda: self.set_exposure(1),
			
			'z': lambda: self.delayed_takeoff(),
			# 'x': lambda: self.toggle_keep_distance()
		}

		controls_keyrelease_QWERTY = {
			'w': lambda: self.set_speed(C_PITCH, 0),
			's': lambda: self.set_speed(C_PITCH, 0),
			'a': lambda: self.set_speed(C_ROLL, 0),
			'd': lambda: self.set_speed(C_ROLL, 0),
			'q': lambda: self.set_speed(C_YAW, 0),
			'e': lambda: self.set_speed(C_YAW, 0),
			'Key.left': lambda: self.set_speed(C_YAW, 0),
			'Key.right': lambda: self.set_speed(C_YAW, 0),
			'Key.up': lambda: self.set_speed(C_THROTTLE, 0),
			'Key.down': lambda: self.set_speed(C_THROTTLE, 0)
		}


		self.controls_keypress = controls_keypress_QWERTY
		self.controls_keyrelease = controls_keyrelease_QWERTY
		self.key_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
		self.key_listener.start()

	def check_pose(self, pose):
		''' Check if we detect a pose in the body detected by PosetNet '''
		try:
			vert_angle_right_arm = vertical_angle(pose.keypoints['right wrist'].xy, pose.keypoints['right elbow'].xy)
		except:
			vert_angle_right_arm = None
		try:
			vert_angle_left_arm = vertical_angle(pose.keypoints['left wrist'].xy, pose.keypoints['left elbow'].xy)
		except:
			vert_angle_left_arm = None
		try:
			left_hand_up = pose.keypoints['left wrist'].xy[1] < pose.keypoints['neck'].xy[1]
		except:
			left_hand_up = None
		try:
			right_hand_up = pose.keypoints['right wrist'].xy[1] < pose.keypoints['neck'].xy[1]
		except:
			right_hand_up = None

		print (right_hand_up, left_hand_up)
		if right_hand_up:
			if not left_hand_up:
				# Only right arm up
				if 'right shoulder' in pose.keypoints and (pose.keypoints['right shoulder'].xy[0]-pose.keypoints['neck'].xy[0])*(pose.keypoints['right wrist'].xy[0]-pose.keypoints['neck'].xy[0])>0:
				# Right shoudler and right hand on the same side
					if vert_angle_right_arm:
						if vert_angle_right_arm < -15:
							return C_RIGHT_ARM_UP_OPEN
						if 15 < vert_angle_right_arm < 90:
							return C_RIGHT_ARM_UP_CLOSED
				elif 'left ear' in pose.keypoints and pose.shoulders_width and distance(pose.keypoints['right wrist'].xy, pose.keypoints['left ear'].xy) < pose.shoulders_width/4:
					# Right hand close to left ear
					return C_RIGHT_HAND_ON_LEFT_EAR
			else:
				# Both hands up
				# Check if both hands are on the ears
				if 'right ear' in pose.keypoints and 'left ear' in pose.keypoints:
					ear_dist = distance(pose.keypoints['right ear'].xy, pose.keypoints['left ear'].xy)
					if distance(pose.keypoints['right wrist'].xy, pose.keypoints['left ear'].xy)<ear_dist/3 and distance(l_wrist,l_ear)<ear_dist/3:
						return C_HANDS_ON_EARS
				# Check if boths hands are closed to each other and above nose
				if pose.shoulders_width and 'nose' in pose.keypoints:
					near_dist = pose.shoulders_width
					if pose.keypoints['nose'].xy[1] > pose.keypoints['right wrist'].xy[1] and distance(pose.keypoints['right wrist'].xy, pose.keypoints['left wrist'].xy) < near_dist:
						return C_CLOSE_HANDS_UP
		else:
			if left_hand_up:
				# Only left arm up
				if 'left shoulder' in pose.keypoints and (pose.keypoints['left shoulder'].xy[0]-pose.keypoints['neck'].xy[0])*(pose.keypoints['left wrist'].xy[0]-pose.keypoints['neck'].xy[0])>0:
					# Left shoudler and left hand on the same side
					if vert_angle_left_arm:
						if vert_angle_left_arm < -15:
							return C_LEFT_ARM_UP_CLOSED
						if 15 < vert_angle_left_arm < 90:
							return C_LEFT_ARM_UP_OPEN
				elif 'right ear' in pose.keypoints and pose.shoulders_width and distance(pose.keypoints['left wrist'].xy,pose.keypoints['right ear'].xy) < self.pose.shoulders_width/4:
					# Left hand close to right ear
					return C_LEFT_HAND_ON_RIGHT_EAR
			else:
				# Both wrists under the neck
				if 'neck' in pose.keypoints and pose.shoulders_width and 'right wrist' in pose.keypoints and 'left wrist' in pose.keypoints:
					near_dist = pose.shoulders_width/3
					if distance(pose.keypoints['right wrist'].xy, pose.keypoints['neck'].xy) < near_dist and distance(pose.keypoints['left wrist'].xy, pose.keypoints['neck'].xy) < near_dist :
						return C_HANDS_ON_NECK
		return None

	def process_frame(self, raw_frame):
		''' Analyze the frame and return the frame with information (HUD, openpose skeleton) drawn on it '''
		frame = raw_frame.copy()
		h,w = (360, 480)
		proximity = int(w/2.6)
		min_distance = int(w/2)
		
		# Is there a scheduled takeoff ?
		if self.scheduled_takeoff and time.time() > self.scheduled_takeoff:
			self.scheduled_takeoff = None
			self.drone.takeoff()

		self.axis_speed = self.cmd_axis_speed.copy()
		# If we are on the point to take a picture, the tracking is temporarily desactivated (2s)
		if self.timestamp_take_picture:
			if time.time() - self.timestamp_take_picture > 2:
				self.timestamp_take_picture = None
				self.drone.take_picture()
		else:
			# If we are doing a 360, where are we in our 360 ?
			if self.yaw_to_consume > 0:
				consumed = self.yaw - self.prev_yaw
				self.prev_yaw = self.yaw
				if consumed < 0: consumed += 360
				self.yaw_consumed += consumed
				if self.yaw_consumed > self.yaw_to_consume:
					self.yaw_to_consume = 0
					self.axis_speed[C_YAW]= 0
				else:
					self.axis_speed[C_YAW]= self.def_speed[C_YAW]

			# We are not flying, we check a potential morse code 
			if not self.is_flying:
				pressing, detected = self.morse.eval(frame)

			# Call to openpose detection
			if self.use_posenet:
				poses = self.op.eval(frame)
				
				target = None
				# Our target is the person whose index is 0 in pose_kps
				self.gesture = None
				if len(poses) > 0 : 
					# We found a body, so we can cancel the exploring 360
					self.yaw_to_consume = 0

					# Do we recognize a predefined gesture ?
					self.pose = poses[0]
					self.gesture = self.check_pose(self.pose)
					if self.gesture is not None:
						# We trigger the associated action
						log.info(f"pose detected : {self.pose}")
						if self.gesture == C_HANDS_ON_NECK or self.gesture == C_HANDS_ON_EARS:
							# Take a picture in 1 second
							if self.timestamp_take_picture is None:
								log.info("Take a picture in 1 second")
								self.timestamp_take_picture = time.time()
								self.sound_player.play("taking picture")
								
	
						elif self.gesture == C_RIGHT_ARM_UP_CLOSED:
							log.info("GOING LEFT from pose")
							self.axis_speed[C_ROLL] = self.def_speed[C_ROLL]
						elif self.gesture == C_RIGHT_ARM_UP_OPEN:
							log.info("GOING RIGHT from pose")
							self.axis_speed[C_ROLL] = -self.def_speed[C_ROLL]
						elif self.gesture == C_LEFT_ARM_UP_CLOSED:
							log.info("GOING FORWARD from pose")
							self.axis_speed[C_PITCH] = self.def_speed[C_PITCH]
						elif self.gesture == C_LEFT_ARM_UP_OPEN:
							log.info("GOING BACKWARD from pose")
							self.axis_speed[C_PITCH] = -self.def_speed[C_PITCH]
						elif self.gesture == C_CLOSE_HANDS_UP:
							# Locked distance mode
							if self.keep_distance is None:
								if  time.time() - self.timestamp_keep_distance > 2:
									# The first frame of a serie to activate the distance keeping
									self.keep_distance = self.pose.shoulders_width
									self.timestamp_keep_distance = time.time()
									log.info(f"KEEP DISTANCE {self.keep_distance}")
									self.pid_pitch = PID(0.5,0.04,0.3,setpoint=0,output_limits=(-50,50))
									#self.graph_distance = RollingGraph(window_name="Distance", y_max=500, threshold=self.keep_distance, waitKey=False)
							else:
								if time.time() - self.timestamp_keep_distance > 2:
									log.info("KEEP DISTANCE FINISHED")
									self.keep_distance = None
									self.timestamp_keep_distance = time.time()
							
						elif self.gesture == C_RIGHT_HAND_ON_LEFT_EAR:
							# Get close to the body then palm landing
							if not self.palm_landing_approach:
								self.palm_landing_approach = True
								self.keep_distance = proximity
								self.timestamp_keep_distance = time.time()
								log.info("APPROACHING on pose")
								self.pid_pitch = PID(0.2,0.02,0.1,setpoint=0,output_limits=(-45,45))
								#self.graph_distance = RollingGraph(window_name="Distance", y_max=500, threshold=self.keep_distance, waitKey=False)
						elif self.gesture == C_LEFT_HAND_ON_RIGHT_EAR:
							if not self.palm_landing:
								log.info("LANDING on pose")
								# Landing
								self.toggle_tracking(tracking=False)
								self.drone.land()	  

					# Draw the skeleton on the frame
					self.pose.draw_body(frame)
					
					# In tracking mode, we track a specific body part (an openpose keypoint):
					# the nose if visible, otherwise the neck, otherwise the midhip
					# The tracker tries to align that body part with the reference point (ref_x, ref_y)
					target = self.pose.get_kp('nose')
					if target is not None:
						ref_x = int(w/2)
						ref_y = int(h*0.35)
					else:
						target = self.pose.get_kp('neck')
						if target is not None:		 
							ref_x = int(w/2)
							ref_y = int(h/2)
						else:
							target = self.pose.get_kp('mid hip')
							if target is not None:		 
								ref_x = int(w/2)
								ref_y = int(0.75*h)
				if self.tracking:
					if target is not None:
						self.body_in_prev_frame = True
						# We draw an arrow from the reference point to the body part we are targeting	   
						h,w = (360, 480)
						xoff = int(target[0]-ref_x)
						yoff = int(ref_y-target[1])
						cv2.circle(frame, (ref_x, ref_y), 15, (250,150,0), 1, cv2.LINE_AA)
						cv2.arrowedLine(frame, (ref_x, ref_y), tuple(target), (250, 150, 0), 6)
					   
						# The PID controllers calculate the new speeds for yaw and throttle
						self.axis_speed[C_YAW]= int(-self.pid_yaw(xoff))
						log.debug(f"xoff: {xoff} - speed_yaw: {self.axis_speed[C_YAW]}")
						self.last_rotation_is_cw = self.axis_speed[C_YAW]> 0

						self.axis_speed[C_THROTTLE] = int(-self.pid_throttle(yoff))
						log.debug(f"yoff: {yoff} - speed_throttle: {self.axis_speed[C_THROTTLE]}")

						# If in locked distance mode
						if self.keep_distance and self.pose.shoulders_width:   
							if self.palm_landing_approach and self.pose.shoulders_width>self.keep_distance:
								# The drone is now close enough to the body
								# Let's do the palm landing
								log.info("PALM LANDING after approaching")
								self.palm_landing_approach = False
								self.toggle_tracking(tracking=False)
								self.palm_land() 
							else:
								self.axis_speed[C_PITCH] = int(self.pid_pitch(self.pose.shoulders_width-self.keep_distance))
								log.debug(f"Target distance: {self.keep_distance} - cur: {self.pose.shoulders_width} -speed_pitch: {self.axis_speed[C_PITCH]}")
					else: # Tracking but no body detected
						if self.body_in_prev_frame:
							self.timestamp_no_body = time.time()
							self.body_in_prev_frame = False
							self.axis_speed[C_THROTTLE] = self.prev_axis_speed[C_THROTTLE]
							self.axis_speed[C_YAW]= self.prev_axis_speed[C_YAW]
						else:
							if time.time() - self.timestamp_no_body < 1:
								print("NO BODY SINCE < 1", self.axis_speed, self.prev_axis_speed)
								self.axis_speed[C_THROTTLE] = self.prev_axis_speed[C_THROTTLE]
								self.axis_speed[C_YAW]= self.prev_axis_speed[C_YAW]
							else:
								log.debug("NO BODY detected for 1s -> rotate")
								self.axis_speed[C_YAW]= self.def_speed[C_YAW]* (1 if self.last_rotation_is_cw else -1)

		# Send axis commands to the drone
		for axis, command in enumerate(self.axis_command):
			if self.axis_speed[axis] is not None and self.axis_speed[axis] != self.prev_axis_speed[axis]:
				# log.debug(f"COMMAND {axis} : {self.axis_speed[axis]}")
				command(self.axis_speed[axis])
				self.prev_axis_speed[axis] = self.axis_speed[axis]
			else:
				# This line is necessary to display current values in 'self.write_hud'
				self.axis_speed[axis] = self.prev_axis_speed[axis]
		
		# Write the HUD on the frame
		frame = self.write_hud(frame)
		return frame

	def write_hud(self, frame):
		''' Draw drone info on frame '''
		class HUD:
			def __init__(self, def_color=(255, 170, 0)):
				self.def_color = def_color
				self.infos = []
			def add(self, info, color=None):
				if color is None: color = self.def_color
				self.infos.append((info, color))
			def draw(self, frame):
				i=0
				for (info, color) in self.infos:
					cv2.putText(frame, info, (0, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) #lineType=30)
					i+=1
				
		hud = HUD()
		hud.add(f"FPS {self.fps.get():.2f}")
		hud.add(f"BAT {self.battery}")
		
		if self.is_flying:
			hud.add("FLYING", (0,255,0))
		else:
			hud.add("NOT FLYING", (0,0,255))
		hud.add(f"TRACKING {'ON' if self.tracking else 'OFF'}", (0,255,0) if self.tracking else (0,0,255) )
		hud.add(f"EXPO {self.exposure}")
		
		if self.axis_speed[C_YAW] > 0:
			hud.add(f"CW {self.axis_speed[C_YAW]}", (0,255,0))
		elif self.axis_speed[C_YAW] < 0:
			hud.add(f"CCW {-self.axis_speed[C_YAW]}", (0,0,255))
		else:
			hud.add(f"CW 0")
		if self.axis_speed[C_ROLL] > 0:
			hud.add(f"RIGHT {self.axis_speed[C_ROLL]}", (0,255,0))
		elif self.axis_speed[C_ROLL] < 0:
			hud.add(f"LEFT {-self.axis_speed[C_ROLL]}", (0,0,255))
		else:
			hud.add(f"RIGHT 0")
		if self.axis_speed[C_PITCH] > 0:
			hud.add(f"FORWARD {self.axis_speed[C_PITCH]}", (0,255,0))
		elif self.axis_speed[C_PITCH] < 0:
			hud.add(f"BACKWARD {-self.axis_speed[C_PITCH]}", (0,0,255))
		else:
			hud.add(f"FORWARD 0")
		if self.axis_speed[C_THROTTLE] > 0:
			hud.add(f"UP {self.axis_speed[C_THROTTLE]}", (0,255,0))
		elif self.axis_speed[C_THROTTLE] < 0:
			hud.add(f"DOWN {-self.axis_speed[C_THROTTLE]}", (0,0,255))
		else:
			hud.add(f"UP 0")

		if self.use_posenet:
			hud.add(f"POSE: {self.gesture}", (0,255,0) if self.gesture else (255, 170, 0))
		
		if self.keep_distance:
			try:
				hud.add(f"Target: {self.keep_distance} - curr: {self.pose.shoulders_width}", (0,255,0))
			except:
				pass
			#if pose.shoulders_width: self.graph_distance.new_iter([pose.shoulders_width])
		if self.timestamp_take_picture: hud.add("Taking a picture", (0,255,0))
		if self.palm_landing:
			hud.add("Palm landing...", (0,255,0))
		if self.palm_landing_approach:
			hud.add("In approach for palm landing...", (0,255,0))
		if self.tracking and not self.body_in_prev_frame and time.time() - self.timestamp_no_body > 0.5:
			hud.add("Searching...", (0,255,0))
		if self.throw_ongoing:
			hud.add("Throw ongoing...", (0,255,0))
		if self.scheduled_takeoff:
			seconds_left = int(self.scheduled_takeoff - time.time())
			hud.add(f"Takeoff in {seconds_left}s")

		hud.draw(frame)
		return frame

	def take_picture(self):
		''' Tell drone to take picture, image sent to file handler '''
		self.drone.take_picture()

	def set_exposure(self, expo):
		''' Change exposure of drone camera '''
		if expo == 0:
			self.exposure = 0
		elif expo == 1:
			self.exposure = min(9, self.exposure+1)
		elif expo == -1:
			self.exposure = max(-9, self.exposure-1)
		self.drone.set_exposure(self.exposure)
		log.info(f"EXPOSURE {self.exposure}")

	def palm_land(self):
		''' Tell drone to land '''
		self.palm_landing = True
		self.drone.palm_land()

	def throw_and_go(self, tracking=False):
		''' Tell drone to start a 'throw and go' '''
		self.drone.throw_and_go()	  
		self.tracking_after_takeoff = tracking
		
	def delayed_takeoff(self, delay=5):
		self.scheduled_takeoff = time.time()+delay
		self.tracking_after_takeoff = False
		#self.keep_distance = True
		''' for debugging '''
		#self.toggle_tracking(True)
		
	def clockwise_degrees(self, degrees):
		self.yaw_to_consume = degrees
		self.yaw_consumed = 0
		self.prev_yaw = self.yaw
		
	def toggle_openpose(self):
		self.use_posenet = not self.use_posenet
		if not self.use_posenet:
			# Desactivate tracking
			self.toggle_tracking(tracking=False)
		log.info('OPENPOSE '+("ON" if self.use_posenet else "OFF"))
 
	def toggle_tracking(self, tracking=None):
		""" 
			If tracking is None, toggle value of self.tracking
			Else self.tracking take the same value as tracking
		"""
		
		if tracking is None:
			self.tracking = not self.tracking
		else:
			self.tracking = tracking
		if self.tracking:
			log.info("ACTIVATE TRACKING")
			# Needs openpose
			self.use_posenet = True
			# Start an explarotary 360
			#self.clockwise_degrees(360)
			# Init a PID controller for the yaw
			self.pid_yaw = PID(0.25,0,0,setpoint=0,output_limits=(-100,100))
			# ... and one for the throttle
			self.pid_throttle = PID(0.4,0,0,setpoint=0,output_limits=(-80,100))
			# self.init_tracking = True
		else:
			self.axis_speed = np.zeros(4, dtype=np.int32)
			self.keep_distance = None
		return
		
	def toggle_keep_distance(self):
		self.keep_distance = self.pose.shoulders_width

	def flight_data_handler(self, event, sender, data):
		''' Listener to flight data from the drone. '''
		self.battery = data.battery_percentage
		self.fly_mode = data.fly_mode
		self.throw_fly_timer = data.throw_fly_timer
		self.throw_ongoing = data.throw_fly_timer > 0
		
		if self.is_flying != data.em_sky:
			self.is_flying = data.em_sky
			log.debug(f"FLYING : {self.is_flying}")
			if not self.is_flying:
				self.reset()
			else:
				if self.tracking_after_takeoff:
					log.debug("Tracking on after takeoff")
					self.toggle_tracking(True)

	def log_data_handler(self, event, sender, data):
		''' Listener to log data from the drone. '''
		pos_x = -data.mvo.pos_x
		pos_y = -data.mvo.pos_y
		pos_z = -data.mvo.pos_z
		if abs(pos_x)+abs(pos_y)+abs(pos_z) > 0.07:
			if self.ref_pos_x == -1: # First time we have meaningful values, we store them as reference
				self.ref_pos_x = pos_x
				self.ref_pos_y = pos_y
				self.ref_pos_z = pos_z
			else:
				self.pos_x = pos_x - self.ref_pos_x
				self.pos_y = pos_y - self.ref_pos_y
				self.pos_z = pos_z - self.ref_pos_z
		
		qx = data.imu.q1
		qy = data.imu.q2
		qz = data.imu.q3
		qw = data.imu.q0
		self.yaw = quat_to_yaw_deg(qx,qy,qz,qw)
		
		if self.write_log_data:
			if self.write_header:
				self.log_file.write('%s\n' % data.format_cvs_header())
				self.write_header = False
			self.log_file.write('%s\n' % data.format_cvs())

	def handle_flight_received(self, event, sender, data):
		''' Create a file in local directory to receive image from the drone '''
		path = f'{self.media_directory}/tello-{datetime.datetime.now().strftime(self.date_fmt)}.jpg' 
		with open(path, 'wb') as out_file:
			out_file.write(data)
		log.info('Saved photo to %s' % path)

if __name__ == '__main__':
	ap=argparse.ArgumentParser()
	ap.add_argument("-l","--log_level", help="select a log level (info, debug)")
	args=ap.parse_args()

	main(log_level=args.log_level)
