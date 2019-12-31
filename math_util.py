import random
import cv2
import numpy as np
from math import atan, atan2, degrees, sqrt, pi, floor

''' all function should accept and return point with (x, y) format '''

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

def quat_to_yaw_deg(qx,qy,qz,qw):
	''' Calculate yaw from quaternion '''
	degree = pi/180
	sqy = qy*qy
	sqz = qz*qz
	siny = 2 * (qw*qz+qx*qy)
	cosy = 1 - 2*(qy*qy+qz*qz)
	yaw = int(atan2(siny,cosy)/degree)
	return yaw

def angle(A, B, C):
	''' Calculate the angle between segment(A,p2) and segment (p2,p3) '''
	return degrees(atan2(C[1]-B[1],C[0]-B[0]) - atan2(A[1]-B[1],A[0]-B[0]))%360

def vertical_angle(A, B): #which one first matter, vertical angle of vector A->B
	AB = B-A
	AB[1] = -AB[1] # flip y
	return degrees(atan2(AB[1], AB[0]))

def distance(A, B):
	return np.linalg.norm(A-B)

def rand_color():
	return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
