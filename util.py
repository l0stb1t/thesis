import os
import cv2
import time
import signal
import ctypes
import random
import pygame
import logging
import numpy as np
import multiprocessing as mp

from math_util import *
from constants import *
from contextlib import suppress

''' all function should accept and return point with (x, y) format '''

''' old to be removed '''
def draw_points(surf, points, top=(0,0), color=C_RED):
	for point in points:
		pygame.draw.circle(surf, color, (point.astype(np.uint32) + top), 3)

def get_boundbox(kps_coord, kps_score):
	min_x = -1
	max_x = -1
	min_y = -1	
	max_y = -1
	
	for i in range(C_NKP):
		if kps_score[i] >= C_KP_THRESHOLD:
			x = kps_coord[i][0]
			y = kps_coord[i][1]
			
			if max_x == -1 or x > max_x:
				max_x = x
			elif min_x == -1 or x < min_x:
				min_x = x
			if max_y == -1 or y > max_y:
				max_y = y
			elif min_y == -1 or y < min_y:
				min_y = y
	
	return (min_x, min_y), (max_x, max_y)

def draw_pose(surf, kps_coord, kps_score, color=C_RED):
	for i in range(18):
		if kps_score[i] >= C_KP_THRESHOLD:
			pygame.draw.circle(surf, color, kps_coord[i], 3)
			
def draw_pose2(surf, kps_coord, kps_score, color=C_RED):
	if kps_score[C_LEYE] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_LEYE], 3)
	if kps_score[C_REYE] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_REYE], 3)
	if kps_score[C_LEAR] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_LEAR], 3)
	if kps_score[C_REAR] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_REAR], 3)
	if kps_score[C_NOSE] >= C_KP_THRESHOLD:
		pygame.draw.circle(surf, color, kps_coord[C_NOSE], 3)
		
	for pair in C_PAIRS:
		if kps_score[pair[0]] >= C_KP_THRESHOLD and kps_score[pair[1]] >= C_KP_THRESHOLD:
			pygame.draw.line(surf, color, kps_coord[pair[0]], kps_coord[pair[1]], 3)

def draw_boundbox(surf, kps_coord, kps_score, color=C_GREEN):
	top, bottom = get_bound_box(kps_coord, kps_score)
	pygame.draw.circle(surf, C_RED, top, 3)
	pygame.draw.circle(surf, C_RED, bottom, 3)
	pygame.draw.rect(surf, color, (top[0], top[1], bottom[0]-top[0], bottom[1]-top[1]), 3)
	return top, bottom
''' old functions end here '''

def init_pygame_window(name='', size=(480,360)):
	pygame.init()
	display = pygame.display.set_mode(size)
	pygame.display.set_caption(name)
	return display

class Pose:
	# __slots__ = ['keypoints', 'score', 'top', 'bottom', 'center', 'color', 'ttl', '_shoulders_width']

	def __init__(self, keypoints, score=None):
		self.keypoints = keypoints
		self.score = score
		
		self.center 			= None # center of the bounding_box
		self.top 				= None # top left
		self.bottom 			= None # bottom right
		self.color 				= None
		self.ttl 				= 5
		self.__kp_cache 		= [None]*C_NKP
		
	def get_kp(self, name):
		try:
			return self.keypoints[name].xy
		except:
			return None	
		
	def has_kp(self, kp_id): #return False/Keypoint
		if self.__kp_cache[kp_id] is None: # we haven't evaluated this yet
			kp_name = C_KP_NAMES[kp_id]
			if  kp_name in self.keypoints:
				self.__kp_cache[kp_id] = self.keypoints[kp_name]
			else:
				self.__kp_cache[kp_id] = False
		return self.__kp_cache[kp_id]
				
	def draw_pose(self, surf, color=C_RED):
		if self.color is not None:
			color = self.color
			
		with suppress(KeyError): pygame.draw.circle(surf, C_BLUE, self.keypoints['left eye'].xy, 3)
		with suppress(KeyError): pygame.draw.circle(surf, C_GREEN, self.keypoints['right eye'].xy, 3)
		with suppress(KeyError): pygame.draw.circle(surf, C_BLUE, self.keypoints['left ear'].xy, 3)
		with suppress(KeyError): pygame.draw.circle(surf, C_GREEN, self.keypoints['right ear'].xy, 3)
		with suppress(KeyError): pygame.draw.circle(surf, color, self.keypoints['nose'].xy, 3)
			
		for pair in C_PAIRS:
			if len(pair) == 3:
				with suppress(KeyError): pygame.draw.line(surf, pair[2], self.keypoints[C_KP_NAMES[pair[0]]].xy, self.keypoints[C_KP_NAMES[pair[1]]].xy, 3)
			else:
				with suppress(KeyError): pygame.draw.line(surf, color, self.keypoints[C_KP_NAMES[pair[0]]].xy, self.keypoints[C_KP_NAMES[pair[1]]].xy, 3)
	def get_boundbox(self):		
		if self.top is None:
			i = iter(self.keypoints)
			x, y = self.keypoints[next(i)].xy
			
			min_x = x
			min_y = y
			max_x = x
			max_y = y
			
			for kp_name in i:
				x, y = self.keypoints[kp_name].xy
				if x > max_x: max_x = x
				elif x < min_x: min_x = x
				
				if y > max_y: max_y = y
				elif y < min_y: min_y = y
				
			self.top 	= np.array((min_x, min_y), dtype=np.uint32)
			self.bottom = np.array((max_x, max_y), dtype=np.uint32)
		return self.top, self.bottom
	
	def get_boundbox_center(self):
		if self.center is None:
			top, bottom = self.get_boundbox()
			self.center = (top+bottom)/2
		return self.center
	
	def center_distance(self, pose):
		return distance(self.get_boundbox_center(), pose.get_boundbox_center())
		
	def generic_distance(self, pose):
		with suppress(KeyError):
			return distance(self.keypoints['nose'].xy, pose.keypoints['nose'].xy)
		with suppress(KeyError):
			return distance(self.keypoints['left eye'].xy, pose.keypoints['left eye'].xy)
		with suppress(KeyError):
			return distance(self.keypoints['right eye'].xy, pose.keypoints['right eye'].xy)
		with suppress(KeyError):
			return distance(self.keypoints['left ear'].xy, pose.keypoints['left ear'].xy)
		with suppress(KeyError):
			return distance(self.keypoints['right ear'].xy, pose.keypoints['right ear'].xy)
		with suppress(KeyError):
			return distance(self.keypoints['neck'].xy, pose.keypoints['neck'].xy)
		return self.center_distance(pose)
			
	def draw_boundbox(self, surf, color=C_GREEN):
		top, bottom = self.get_boundbox()
		pygame.draw.circle(surf, C_RED, top, 3)
		pygame.draw.circle(surf, C_RED, bottom, 3)
		pygame.draw.rect(surf, color, (top[0], top[1], bottom[0]-top[0], bottom[1]-top[1]), 3) #x, y, width, height
		
	''' to be removed in the future '''
	def draw_body(self, frame):
		points = []
		for keypoint in self.keypoints.values():
			points.append(cv2.KeyPoint(keypoint.xy[0], keypoint.xy[1], 5))
		cv2.drawKeypoints(frame, points, outImage=frame, color=C_BLUE)
	
	def __repr__(self):
		return 'Pose({}, {})'.format(self.keypoints, self.score)
		
class Keypoint:
	__slots__ = ['name', 'xy', 'score']

	def __init__(self, name, xy, score=None):
		self.name = name
		self.xy = xy
		self.score = score
		
	def __repr__(self):
		return 'Keypoint(<{}>, {}, {})'.format(self.name, self.xy, self.score)
		
class Analyzer:
	def __init__(self, pose=None):
		self.pose = pose
		
		self.__g_lhu = None
		self.__g_rhu = None
		self.__g_shoulders_vert_angle = None
		self.__g_shoulders_vert_angle2 = None
		self.__g_shoulders_width = None
		self.__g_lshoulder_width = None
		self.__g_rshoulder_width = None
		self.__g_eyes_distance = None
		self.__g_rotation = None
		self.__g_standing = None
		self.__g_sitting = None
		self.__g_lying = None
		self.__frontal_face_boundingbox = None
	
	def feed(self, pose):
		self.__init__(pose)
	
	@property
	def g_standing(self): #return true if standing, false if no or unsure
		if self.__g_standing is None:
			left_leg_vert_angle = self.vertical_angle(C_LHIP, C_LKNEE)
			right_leg_vert_agnle = self.vertical_angle(C_RHIP, C_RKNEE)
						
			for e in (left_leg_vert_angle, right_leg_vert_agnle):
				if e and (e<=-70 and e >=-110):
					self.__g_standing = True
				else:
					self.__g_standing = False
					break
		return self.__g_standing
		
	@property
	def g_sitting(self): #return true if standing, false if no or unsure
		if self.__g_sitting is None:
			left_leg_vert_angle = self.vertical_angle(C_LHIP, C_LKNEE)
			right_leg_vert_agnle = self.vertical_angle(C_RHIP, C_RKNEE)
						
			for e in (left_leg_vert_angle, right_leg_vert_agnle):
				if e and (not (e<=-70 and e >=-110)):
					self.__g_sitting = True
				else:
					self.__g_sitting = False
					break
		return self.__g_sitting
	
	@property
	def g_upper_height(self):
		lshoulder = self.pose.has_kp(C_LSHOULDER)
		if not lshoulder:
			d1=0
		else:
			lhip = self.pose.has_kp(C_LHIP)
			if not lhip:
				d1=0
			else:
				d1 = lhip.xy[1]-lshoulder.xy[1]
				
		rshoulder = self.pose.has_kp(C_RSHOULDER)
		if not rshoulder:
			d2=0
		else:
			rhip = self.pose.has_kp(C_RHIP)
			if not rhip:
				d2=0
			else:
				d2=rhip.xy[1]-rshoulder.xy[1]
		r = max(d1, d2)
		if r <=0:
			return None
		return r
			
	
	@property
	def g_lower_height(self):
		lhip = self.pose.has_kp(C_LHIP)
		if not lhip:
			d1 = 0
		else:
			lankle = self.pose.has_kp(C_LANKLE)
			if not lankle:
				d1 = 0
			else:
				d1 = lankle.xy[1]-lhip.xy[1]
		
		rhip = self.pose.has_kp(C_RHIP)
		if not rhip:
			d2 = 0
		else:
			rankle = self.pose.has_kp(C_RANKLE)
			if not rankle:
				d2 = 0
			else:
				d2 = rankle.xy[1]-rhip.xy[1]
		r = max(d1, d2)
		if r <=0:
			return None
		return r
			
	@property
	def g_sitting2(self):
		d1 = self.g_upper_height
		d2 = self.g_lower_height
		
		if d1==None or d2==None:
			return None
		if (d1/d2)>0.8:
			return True
		else:
			return False
	
	@property
	def g_standing2(self):
		d1 = self.g_upper_height
		d2 = self.g_lower_height
		
		if d1==None or d2==None:
			return None
		# print ((d1/d2)*100)
		if (d1/d2)<=0.8:
			return True
		else:
			return False
		
	
	@property
	def g_lying(self):
		if self.__g_lying is None:
			top, bottom = self.pose.get_boundbox()
			width = bottom[0] - top[0]
			height = bottom[1] - top[1]
			
			if height/width<=0.5:
				self.__g_lying = True
			else:
				self.__g_lying = False
		return self.__g_lying 
	
	@property
	def g_shoulders_width(self): # return width/None 
		if self.__g_shoulders_width is None:
			self.__g_shoulders_width = self.distance(C_LSHOULDER, C_RSHOULDER)
		return self.__g_shoulders_width
	
	@property
	def g_eyes_distance(self):
		if self.__g_eyes_distance is None:
			self.__g_eyes_distance = self.distance(C_LEYE, C_REYE)
		return self.__g_eyes_distance
			
	@property
	def g_lshoulder_width(self):
		if self.__g_lshoulder_width is None:
			self.__g_lshoulder_width = self.distance(C_LSHOULDER, C_NECK)
		return self.__g_lshoulder_width
		
	@property
	def g_rotation(self):
		if self.__g_rotation is None:
			d1 = self.distance(C_NOSE, C_LEAR)
			d2 = self.distance(C_NOSE, C_REAR)
		
			if d1 is None:
				return None
			if d2 is None:
				return None
			self.__g_rotation = (d1-d2)/(d1+d2)
		return self.__g_rotation
	
	@property
	def g_vrotation(self):
		v1 = self.vertical_angle(C_NOSE, C_LEAR)
		v2 = self.vertical_angle(C_NOSE, C_REAR)
		
		if v1 is None or v2 is None:
			return None
		t=v1-v2
		if t<0:
			t+=180
		elif t>0:
			t-=180
		return t
	
	@property
	def g_rshoulder_width(self):
		if self.__g_rshoulder_width is None:
			self.__g_rshoulder_width = self.distance(C_RSHOULDER, C_NECK)
		return self.__g_rshoulder_width	
	
	@property
	def g_lhu(self): # gadget left hand up
		if self.__g_lhu is None: # we haven't evaluated this gadget yet
			lwrist = self.pose.has_kp(C_LWRIST)
			neck = self.pose.has_kp(C_NECK)
			if neck and lwrist:
				self.__g_lhu = neck.xy[1] > lwrist.xy[1] #lwrist is higher than neck
			else:
				self.__g_lhu = False
		return self.__g_lhu
	
	@property
	def g_rhu(self): # gadget right hand up
		if self.__g_rhu is None: # we haven't evaluated this gadget yet
			rwrist = self.pose.has_kp(C_RWRIST)
			neck = self.pose.has_kp(C_NECK)
			if neck and rwrist:
				self.__g_rhu = neck.xy[1] > rwrist.xy[1] #lwrist is higher than neck
			else:
				self.__g_rhu = False
		return self.__g_rhu
	
	@property
	def g_shoulders_vert_angle(self):
		if self.__g_shoulders_vert_angle is None:
			self.__g_shoulders_vert_angle = self.vertical_angle(C_LSHOULDER, C_RSHOULDER)
		return self.__g_shoulders_vert_angle
	
	@property
	def g_shoulders_vert_angle2(self):
		''' convert from -180->180 to 0->360 set point is 180 '''
		if self.__g_shoulders_vert_angle2 is None:
			try:
				if self.g_shoulders_vert_angle<0:
					self.__g_shoulders_vert_angle2 = 360 + self.g_shoulders_vert_angle
				else:
					self.__g_shoulders_vert_angle2 = self.g_shoulders_vert_angle
			except:
				pass
		return self.__g_shoulders_vert_angle2
	
	def get_frontal_face_boundingbox(self):
		if self.__frontal_face_boundingbox is None:
			rotation = self.g_rotation
			vrotation = self.g_vrotation
			if rotation is None or vrotation is None:
				return None
			if rotation<-0.5 or rotation>0.5:
				return None
			if vrotation<-40 or vrotation>40:
				return None	
			top_x		= self.pose.has_kp(C_REAR).xy[0]
			bottom_x	= self.pose.has_kp(C_LEAR).xy[0]
			
			width=bottom_x-top_x
			nose = self.pose.has_kp(C_NOSE)
					
			top_y 		= nose.xy[1]-width/2
			''' upper part of face is occluded '''
			if top_y<0:
				return None
			bottom_y 	= nose.xy[1]+width/2
			self.__frontal_face_boundingbox = ((int(top_x), int(top_y)), (int(bottom_x), int(bottom_y)))
		return self.__frontal_face_boundingbox
		
	def draw_frontal_face_boundingbox(self, surf, color=C_GREEN):
		try:
			top, bottom = self.get_frontal_face_boundingbox()
		except:
			return
		pygame.draw.circle(surf, C_RED, top, 3)
		pygame.draw.circle(surf, C_RED, bottom, 3)
		pygame.draw.rect(surf, color, (top[0], top[1], bottom[0]-top[0], bottom[1]-top[1]), 3) #x, y, width, height
	
	def vertical_angle(self, kp_id1, kp_id2): # calcuate vertical angle between 2 point
		kp1 = self.pose.has_kp(kp_id1)
		kp2 = self.pose.has_kp(kp_id2)
		
		if kp1 and kp2:
			return vertical_angle(kp1.xy, kp2.xy)
		return None
		
	def angle(self, A_id, B_id, C_id):
		A = self.pose.has_kp(A_id)
		B = self.pose.has_kp(B_id)
		C = self.pose.has_kp(C_id)
		
		if A and B and C:
			return angle(A.xy, B.xy, C.xy)
		return None
		
	def distance(self, kp_id1, kp_id2):
		kp1 = self.pose.has_kp(kp_id1)
		kp2 = self.pose.has_kp(kp_id2)
		
		if kp1 and kp2:
			return distance(kp1.xy, kp2.xy)
		return None
		
	def simple_gesture(self):
		''' Check if we detect a pose in the body detected by PosetNet '''
		
		vert_angle_right_arm = self.vertical_angle(C_RELBOW, C_RWRIST)
		vert_angle_left_arm = self.vertical_angle(C_LELBOW, C_LWRIST)
	
		left_hand_up = self.g_lhu
		right_hand_up = self.g_rhu
		
		if right_hand_up:
			if not left_hand_up:
				# Only right arm up
				rshoulder 	= self.pose.has_kp(C_RSHOULDER)
				neck 		= self.pose.has_kp(C_NECK)
				rwrist 		= self.pose.has_kp(C_RWRIST)
				if rshoulder and (rshoulder.xy[0]-neck.xy[0])*(rwrist.xy[0]-neck.xy[0])>0:
				# Right shoudler and right hand on the same side
					if vert_angle_right_arm:
						if vert_angle_right_arm > 90:
							return C_RIGHT_ARM_UP_OPEN
						if vert_angle_right_arm < 90:
							return C_RIGHT_ARM_UP_CLOSED
				elif self.pose.has_kp(C_LEAR) and self.g_shoulders_width and self.distance(C_RWRIST, C_LEAR)<self.g_shoulders_width/4:
					# Right hand close to left ear
					return C_RIGHT_HAND_ON_LEFT_EAR
			else:
				# Both hands up
				# Check if both hands are on the ears
				ear_dist = self.distance(C_REAR, C_LEAR)
				if ear_dist:
					if self.distance(C_RWRIST, C_REAR)<ear_dist/3 and self.distance(C_LWRIST,C_LEAR)<ear_dist/3:
						return C_HANDS_ON_EARS
				# Check if boths hands are closed to each other and above nose
				nose = self.pose.has_kp(C_NOSE)
				if self.g_shoulders_width and nose:
					near_dist 	= self.g_shoulders_width
					rwrist 		= self.pose.has_kp(C_RWRIST)
					lwrist 		= self.pose.has_kp(C_LWRIST)
					if nose.xy[1] > rwrist.xy[1] and distance(rwrist.xy, lwrist.xy)<near_dist:
						return C_CLOSE_HANDS_UP
		else:
			if left_hand_up:
				# Only left arm up
				lshoulder 	= self.pose.has_kp(C_LSHOULDER)
				neck 	  	= self.pose.has_kp(C_NECK)
				lwrist 		= self.pose.has_kp(C_LWRIST)
				if lshoulder and (lshoulder.xy[0]-neck.xy[0])*(lwrist.xy[0]-neck.xy[0])>0:
					# Left shoudler and left hand on the same side
					if vert_angle_left_arm:
						if vert_angle_left_arm > 90:
							return C_LEFT_ARM_UP_CLOSED
						if vert_angle_left_arm < 90:
							return C_LEFT_ARM_UP_OPEN
				elif self.pose.has_kp(C_REAR) and self.g_shoulders_width and self.distance(C_LWRIST, C_REAR)<self.g_shoulders_width/4:
					# Left hand close to right ear
					return C_LEFT_HAND_ON_RIGHT_EAR
			else:
				# Both wrists under the neck
				neck = self.pose.has_kp(C_NECK)
				lwrist = self.pose.has_kp(C_LWRIST)
				rwrist = self.pose.has_kp(C_RWRIST)
				if neck and self.g_shoulders_width and lwrist and rwrist:
					near_dist = self.g_shoulders_width/2
					if distance(rwrist.xy, neck.xy)<near_dist and distance(lwrist.xy, neck.xy)<near_dist :
						return C_HANDS_ON_NECK
		return None

''' Analyzer action for a single person '''
class StreamAnalyzer:
	def __init__(self):
		self.first_frame = True
		self.ana = Analyzer()
		self.__prev_gesture = None
		self.__gesture_score = 0
		
	def feed(self, pose):
		self.pose = pose
		self.ana.feed(self.pose)
		gesture = self.ana.simple_gesture()
		
		# if self.first_frame:
		# 	self.first_frame = False
		
		''' inconsistent gesture -> noise '''	
		if self.__prev_gesture != gesture:
			self.__gesture_score = 0
		else:
			if self.__gesture_score < 10:
				self.__gesture_score += 1
		self.__prev_gesture = gesture
		
	def simple_gesture(self):
		if self.__gesture_score >= 10:
			return self.__prev_gesture
		else:
			return C_NOTSURE
			
''' This class tries to track every poses in the prev frame '''
class Tracker:
	def __init__(self):
		self.__first_frame = True
		self.__poses = None
		self.__prev_poses = None
		
	def reset(self):
		self.__first_frame = True
		self.__prev_poses = None
		self.__poses = None
	
	@property
	def first_frame(self):
		return self.__first_frame
	
	''' match {cur_idx: (prev_idx, distacne) '''
	def feed(self, poses):
		# no pose detected, reset the tracker
		if len(poses) == 0:
			self.reset()
			return None
		
		# this is the first frame
		if self.__first_frame: 
			self.__prev_poses = poses
			# assign color id to each pose
			for pose in self.__prev_poses:
				pose.color = rand_color()
			self.__first_frame = False
			return
	
		# build distance table
		# table[i][j] = generic_distance(
		self.__poses = poses
		table = []	
		for prev_pose_idx, prev_pose in enumerate(self.__prev_poses):
			# distances
			ds = []
			for pose_idx, pose in enumerate(self.__poses):
				ds.append( (pose_idx, prev_pose.generic_distance(pose)) )
			ds.sort(key=lambda e: e[1], reverse=True)
			table.append(ds)
		
		# find a match for self.__prev_poses[i]
		def find_match(prev_pose_idx, table, match):			
			pose_idx, distance = table[prev_pose_idx][-1]
			if pose_idx in match and match[pose_idx][0] != prev_pose_idx:
				if distance < match[pose_idx][1]:
					# conflict happened
					conflict_prev_pose_idx = match[pose_idx][0]
					# uppdate new prev_pose_idx into match dictionary
					match[pose_idx] = (prev_pose_idx, distance)
					# choose the next smallest distance
					table[conflict_prev_pose_idx].pop()
					if len(table[conflict_prev_pose_idx]) == 0:
						# gave up, can't find any match
						return 
					find_match(conflict_prev_pose_idx, table, match)
				else:
					# conflict but we have bigger distance, choose the next smallest distance
					table[prev_pose_idx].pop()
					if len(table[prev_pose_idx]) == 0:
						# gave up, cant find any match
						return
					find_match(prev_pose_idx, table, match)
			else:
				# no conflict, we are good to go
				match[pose_idx] = (prev_pose_idx, distance)
				
		match = {}
		for prev_pose_idx in range(len(table)):
			find_match(prev_pose_idx, table, match)
		
		# update color to matched pair	
		new_prev_poses = []
		for pose_idx in match:
			prev_pose_idx = match[pose_idx][0]
			self.__poses[pose_idx].color = self.__prev_poses[prev_pose_idx].color
			new_prev_poses.append(self.__poses[pose_idx])
			
		# a new pose that doesn't match any prev_pose
		for pose_idx, pose in enumerate(self.__poses):
			if pose_idx not in match:
				pose.color = rand_color()
				new_prev_poses.append(pose)
		self.__prev_poses = new_prev_poses
		return match

''' This class track only one specific pose in the prev frame '''
class SingleTracker:
	def __init__(self):
		self.target_pose = None
		self.first_frame = True
		
	def reset(self):
		self.__init__()
	
	''' return None/pose_idx '''
	def feed(self, poses, pose_idx=0):
		''' This is the first frame, choose a target pose '''
		if self.first_frame:
			self.first_frame = False
			self.target_pose = poses[pose_idx]
			return
		
		if len(poses) == 0:
			return None
		
		min_idx = None
		min_distance = 10000
		for pose_idx, pose in enumerate(poses):
			d = self.target_pose.generic_distance(pose)
			if d < min_distance:
				min_idx = pose_idx
				min_distance = d
		''' TODO: add a threshold '''
		self.target_pose = poses[min_idx]
		return min_idx
		
class FPS: # To measure the number of frame per second
	def __init__(self):
		self.frame_count 	= 0
		self.start_time 	= 0
		self.fps 			= 0

	def update(self):
		# we update FPS every 10 frame
		if self.frame_count % 10 == 0:
			if self.start_time != 0:
				self.stop_time 		= time.time()
				self.fps 			= 10/(self.stop_time-self.start_time)
				self.start_time 	= self.stop_time
				self.frame_count 	= 0
			else :
				self.start_time=time.time()	
		self.frame_count += 1
	
	def get(self):
		return self.fps

	def display(self, surf, color=C_GREEN):
		global FONT
		
		text = FONT.render('FPS: %f' % self.get(), False, color, None)
		surf.blit(text, (0, 0))

''' doing face recognitionn in a seperate process '''
class FaceRecognition:
	def __init__(self, face_signature_path='/home/pi/duy.sig', log_level=logging.DEBUG):
		''' this event will be passed to the child process '''
		self.__event = mp.Event()
		self.__face_signature_path = face_signature_path
		
		self.__running 			= mp.sharedctypes.RawValue(ctypes.c_ubyte, 1)
		self.__face_count		= mp.sharedctypes.RawValue(ctypes.c_ushort)
		self.__boundingboxes	= mp.sharedctypes.RawValue((ctypes.c_ushort*4)*10)
		self.__results			= mp.sharedctypes.RawValue(ctypes.c_bool*10)
		self.__face_framebuffer = mp.sharedctypes.RawValue(ctypes.c_ubyte*3*480*360)
		self.__waiting_result 	= False
		
		self.logger = logging.getLogger('FaceRecognition')
		self.logger.setLevel(log_level)
		
		self.__worker = mp.Process(target=self.face_recognition_worker)
		self.__worker.start()
	
	@property
	def waiting_result(self):
		return self.__waiting_result
	
	def has_result(self):
		''' if haven't fed data in or worker is working '''
		if not self.__waiting_result:
			raise Exception('FaceRecognition, call feed first') 
		if self.__event.is_set():
			return None
		
		r = []
		for i in range(self.__face_count.value):
			r.append(self.__results[i])
		self.__waiting_result = False
		return r
	
	def stop(self):
		os.kill(self.__worker.pid, signal.SIG_KILL)
	
	''' return True if worker is not busy, false otherwise '''
	def feed(self, boundingboxes, frame):
		if not self.waiting_result:
			self.__face_count.value = len(boundingboxes)
			for i in range(self.__face_count.value):
				boundingbox = boundingboxes[i]
				self.__boundingboxes[i] = (boundingbox[0][1], boundingbox[1][0], boundingbox[1][1], boundingbox[0][0])
			self.__face_framebuffer[:] = np.ctypeslib.as_ctypes(frame)
			self.__waiting_result = True
			self.__event.set()
			return True
		return False
	
	def face_recognition_worker(self):
		import face_recognition, pickle

		with open(self.__face_signature_path, 'rb') as f:
			known_sig = pickle.loads(f.read())
		self.logger.info('face signature loaded')
		
		while True:
			if not self.__event.wait(timeout=2):
				continue
			face_frame = np.ctypeslib.as_array(self.__face_framebuffer).copy()
			nfaces = self.__face_count.value
			face_boundingboxes = []
			for i in range(nfaces):
				face_boundingboxes.append(tuple(self.__boundingboxes[i]))
			
			encodings = face_recognition.face_encodings(face_frame, known_face_locations=face_boundingboxes)
			results = face_recognition.compare_faces(encodings, known_sig, tolerance=0.4)
			self.logger.debug('face recognition result %s' % repr(results))
			for i in range(nfaces):
				try:
					self.__results[i] = results[i]
				except:
					self.__results[i] = False
			self.__event.clear()
		
