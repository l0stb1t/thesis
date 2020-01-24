import sys
sys.path.insert(0, '/home/pi/thesis')
from util import *
from constants import *
from drone_constants import *

import traceback
import numpy as np
import av, cv2, tellopy, ctypes

from math import pi, atan2, degrees, sqrt
import os, time, datetime, re, logging, argparse
import multiprocessing as mp
from multiprocessing import Process, sharedctypes

from simple_pid import PID
from pose_engine import PoseEngine

from CameraMorse import CameraMorse, RollingGraph
from PN import *

LOG = logging.getLogger("TellPoseNet")
LOG.setLevel(logging.CRITICAL)
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
	global FRAMEBUFFER, RUNNING
	
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	writer = cv2.VideoWriter('/home/pi/exfat/record.avi', fourcc, 20, (480, 360))
	while RUNNING:
		frame = np.ctypeslib.as_array(FRAMEBUFFER).copy()
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		writer.write(frame)

def worker(sockfd, lock):
	global FRAMEBUFFER, RUNNING
	
	display = init_pygame_window('tello')
	drone 	= tellopy.Tello(sockfd=sockfd, no_video_thread=True)
	drone.connect()
	tello = TelloController(drone)
	
	while RUNNING:
		surf, frame = get_surf(lock)
		
		tello.fps.update()
		tello.fps.display(surf)
		
		try:
			tello.process_frame(surf, frame)
			pygame.draw.circle(surf, C_RED, (int(C_WIDTH/2), int(C_HEIGHT/2)), 5)
		except:
			tello.stay()
			traceback.print_exc()
			RUNNING.value = 0
			break
			
		display.blit(surf, (0, 0))
		pygame.display.update()
		pygame.time.delay(5)
		
		key_pressed  = []
		key_released = []
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				RUNNING.value = 0
				break
			elif event.type == pygame.KEYDOWN:
				key_pressed.append(event.key)
			elif event.type == pygame.KEYUP:
				key_released.append(event.key)
		tello.key_pressed  = key_pressed
		tello.key_released = key_released
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
	
	RUNNING		= sharedctypes.RawValue(ctypes.c_ubyte, 1)
	FRAMEBUFFER = sharedctypes.RawValue(ctypes.c_ubyte*3*480*360)
	
	drone = init_drone()
	
	# p_recorder = Process(target=recorder)
	# p_recorder.start()
	
	lock 		= mp.Lock()
	p_worker 	= Process(target=worker, args=(drone.sock.fileno(), lock))
	p_worker.start()
	
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
	
class TelloController(object):
	'''TelloController builds keyboard controls on top of TelloPy as well as generating images from the video stream and enabling opencv support'''

	def __init__(self, drone, media_directory='media'):			
		self.drone 			= drone
		self.debug 			= False
				
		# Flight data
		self.is_flying 		 = False
		self.battery 		 = None
		self.fly_mode 		 = None
		self.throw_fly_timer = 0
		
		self.no_pose_counter = 0
		
		self.speed_vect 	 = np.zeros(4, dtype=np.int32)	 
				   
		self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, 		self.flight_data_handler)
		self.drone.subscribe(self.drone.EVENT_LOG_DATA, 		self.log_data_handler)
		self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED, 	self.handle_flight_received)
		
		self.init_controls()
		
		# Setup PoseNet
		self.pn = PN()
		#self.tracker = Tracker()
		self.single_tracker = SingleTracker()
		
		self.ana = Analyzer()
		
		self.fps 		= FPS()
		self.exposure 	= 0
		
		self.reset()
		self.media_directory = media_directory
		if not os.path.isdir(self.media_directory):
			os.makedirs(self.media_directory)
		
	def set_video_encoder_rate(self, rate):
		self.drone.set_video_encoder_rate(rate)
		self.video_encoder_rate = rate

	def reset(self):
		''' Reset global variables before a fly '''
		LOG.info("RESET")
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

	def exit(self):
		self.toggle_tracking(False)
		self.drone.land()
		self.drone.quit()

	def set_speed(self, axis, speed):
		LOG.debug(f"set speed {axis} {speed}")
		self.cmd_axis_speed[axis] = speed

	def init_controls(self):
		''' Define keys and add listener '''
		
		self.controls_keypress = {
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
			pygame.K_ESCAPE: self.exit,
			
			pygame.K_p: lambda: self.palm_land(),
			pygame.K_t: lambda: self.toggle_tracking(),
			pygame.K_o: lambda: self.toggle_posenet(),
			pygame.K_c: lambda: self.clockwise_degrees(360),
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

		self.controls_keyrelease = {
			pygame.K_w: lambda: self.drone.forward(0),
			pygame.K_s: lambda: self.drone.forward(0),
			pygame.K_a: lambda: self.drone.right(0),
			pygame.K_d: lambda: self.drone.right(0),
			
			pygame.K_LEFT: 	lambda: self.drone.clockwise(0),
			pygame.K_RIGHT: lambda: self.drone.clockwise(0),
			pygame.K_UP: 	lambda: self.drone.up(0),
			pygame.K_DOWN: 	lambda: self.drone.up(0)
		}
		
		key_pressed 	= []
		key_released 	= []
		self.is_pressed = {}
		for key in self.controls_keypress:
			self.is_pressed[key] = False
		
	def write_hud(self):
		pass
	
	def handle_keyboard(self):
		try:
			if len(self.key_pressed):
				for key in self.key_pressed:
					self.is_pressed[key] = True
					if key in self.controls_keypress:
						self.controls_keypress[key]()
		
			if len(self.key_released):
				for key in self.key_released:
					self.is_pressed[key] = False
					if key in self.controls_keyrelease:
						self.controls_keyrelease[key]()
					
			if True in self.is_pressed.values():
				return True
			return False
		except AttributeError:
			# first frame
			return False
				
	def process_frame(self, surf, frame):
		''' if a key is pressed we ignore everything '''
		if self.handle_keyboard():
			self.keep_distance = None
			return
		
		poses = self.pn.eval(frame)
		# poses = []
		if self.tracking:
			''' we found some poses '''
			if len(poses):
				self.no_pose_counter = 0
				target_pose = None
				
				''' we haven't tracked anyone yet'''
				if self.single_tracker.first_frame:
					''' choose a random pose to track '''
					target_pose = poses[0]
					
					''' id is the index in poses array'''
					self.track_id = 0
					self.single_tracker.feed(poses, self.track_id)
				else:
					match_idx = self.single_tracker.feed(poses)
					if match_idx is not None:
						target_pose = poses[match_idx]
				
				if target_pose:
					self.ana.feed(target_pose)
					gesture = self.ana.simple_gesture()
					
					if gesture == C_CLOSE_HANDS_UP:
						print ('C_CLOSE_HANDS_UP')
						self.exit()
					#if gesture == C_RIGHT_ARM_UP_CLOSED:
					# 	self.toggle_tracking()
					
					if (self.single_tracker.first_frame or self.keep_distance is None) and self.ana.g_shoulders_width:
						self.keep_distance = self.ana.g_shoulders_width
					
					target_pose.draw_pose(surf)

					kp_order = (C_NOSE, C_NECK, C_MIDHIP)
					for kp_id in kp_order:
						kp = target_pose.has_kp(kp_id)
						if kp:
							target_kp = kp
							
							if kp_id == C_NOSE:
								center_x = int(C_WIDTH*0.5)
								center_y = int(C_HEIGHT*0.35)
							elif kp_id == C_NECK:
								center_x = int(C_WIDTH*0.5)
								center_y = int(C_HEIGHT*0.5)
							elif kp_id == C_MIDHIP:
								center_x = int(C_WIDTH*0.5)
								center_y = int(C_HEIGHT*0.75)
							break	
																			
					pygame.draw.circle(surf, C_BLUE, (center_x, center_y), 3)
					pygame.draw.line(surf, C_GREEN, (center_x, center_y), target_kp.xy, 3)
					
					x_offset = center_x - target_kp.xy[0]
					y_offset = target_kp.xy[1] - center_y
					
					self.speed_vect[C_YAW] 		= self.pid_yaw(x_offset) 
					self.speed_vect[C_THROTTLE] = self.pid_throttle(y_offset)
					
					
					
					if self.ana.g_shoulders_width and self.keep_distance:		
						d_offset = self.keep_distance - self.ana.g_shoulders_width
						self.speed_vect[C_PITCH] = -self.pid_pitch(d_offset) 
					else:
						self.speed_vect[C_PITCH] = 0
				
					'''
					if self.ana.g_shoulders_vert_angle2 is not None:
						pass
						# self.speed_vect[C_ROLL] = -self.pid_roll(self.ana.g_shoulders_vert_angle2)
					else:
						self.speed_vect[C_ROLL] = 0
					'''
					
					if self.ana.g_rotation is not None:
						rotation = self.ana.g_rotation
					try:
						print ('rotation:', rotation)
						self.speed_vect[C_ROLL] = self.pid_roll(rotation)
					except:
						pass
			else:
				print ('no_pose_counter', self.no_pose_counter)
				self.no_pose_counter += 1
				if self.no_pose_counter > 40:
					self.speed_vect[C_PITCH] = 0
					self.speed_vect[C_THROTTLE] = 0
			self.exec()
		else:
			''' manual control mode '''
			if len(poses):
				target_pose = poses[0]
				target_pose.draw_pose(surf)
				self.ana.feed(target_pose)
				# print (self.ana.g_shoulders_vert_angle)

	def stay(self):
		for i in range(4):
			self.speed_vect[i] = 0
			
	def exec(self):
		pass
		self.drone.up(self.speed_vect[C_THROTTLE])
		self.drone.forward(self.speed_vect[C_PITCH])
		self.drone.clockwise(self.speed_vect[C_YAW])
		self.drone.right(self.speed_vect[C_ROLL])
		
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
		LOG.info(f"EXPOSURE {self.exposure}")

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
		 
	def toggle_tracking(self, tracking=None):		
		if tracking is None:
			self.tracking = not self.tracking
		else:
			self.tracking = tracking
			
		if self.tracking:
			LOG.info("ACTIVATE TRACKING")
			self.pid_roll 		= PID(C_ROLL_KP, 		C_ROLL_KI, 		C_ROLL_KD, 		setpoint=0, output_limits=(-50, 50))
			self.pid_yaw 		= PID(C_YAW_KP, 		C_YAW_KI, 		C_YAW_KD, 		setpoint=0, output_limits=(-100, 100))
			self.pid_pitch		= PID(C_PITCH_KP, 		C_PITCH_KI, 	C_PITCH_KD, 	setpoint=0, output_limits=(-100, 100))
			self.pid_throttle 	= PID(C_THROTTLE_KP, 	C_THROTTLE_KI, 	C_THROTTLE_KD, 	setpoint=0, output_limits=(-100, 100))
		else:
			''' tracking turned off we stop the drone '''
			self.single_tracker.reset()
			self.keep_distance = None
			self.stay()
		return

	def flight_data_handler(self, event, sender, data):
		''' Listener to flight data from the drone. '''
		self.battery = data.battery_percentage
		if (self.battery < 15):
			LOG.critical('Batter low')
		
		if self.is_flying != data.em_sky:
			self.is_flying = data.em_sky
			LOG.debug('Flying %d' % self.is_flying)
			if not self.is_flying:
				self.reset()
				
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
		
	def handle_flight_received(self, event, sender, data):
		''' Create a file in local directory to receive image from the drone '''
		date_fmt = '%Y-%m-%d_%H%M%S'
		path = f'{self.media_directory}/tello-{datetime.datetime.now().strftime(date_fmt)}.jpg' 
		with open(path, 'wb') as out_file:
			out_file.write(data)
		LOG.info('Saved photo to %s' % path)

if __name__ == '__main__':
	ap=argparse.ArgumentParser()
	args=ap.parse_args()

	main()
