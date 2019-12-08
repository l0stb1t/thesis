import pygame
import cv2

C_NECK = 17
C_NOSE = 0
C_LEYE = 1
C_REYE = 2
C_LHIP = 11
C_RHIP = 12
C_LKNEE = 13
C_RKNEE = 14
C_LANKLE = 15
C_RANKLE = 16
C_RWRIST = 10
C_LWRIST = 9
C_RELBOW = 8
C_LELBOW = 7
C_RSHOULDER = 6
C_LSHOULDER = 5
C_REAR = 4
C_LEAR = 3

C_NKP 		= 18
C_MAXPOSE 	= 10
C_KP_THRESHOLD = 0.4
C_PSCORE_THRESHOLD = 0.3

C_RED = (255, 0, 0)
C_GREEN = (0, 255, 0)
C_BLUE = (0, 0, 255)
C_YELLOW = (255, 255, 0)
C_COLORS = (C_RED, C_GREEN, C_BLUE)

C_RIGHT_ARM_UP_OPEN 			= 0
C_RIGHT_ARM_UP_CLOSED 			= 1
C_RIGHT_HAND_ON_LEFT_EAR 		= 2
C_LEFT_ARM_UP_OPEN 				= 3
C_LEFT_ARM_UP_CLOSED 			= 4
C_LEFT_HAND_ON_RIGHT_EAR 		= 5
C_HANDS_ON_EARS 				= 6
C_CLOSE_HANDS_UP 				= 7
C_HANDS_ON_NECK 				= 8


C_YAW 	= 0
C_PITCH = 1
C_ROLL 	= 2
C_THROTTLE = 3

pygame.font.init()
myfont = pygame.font.SysFont(None, 20)
LABELS = [None,]*10

POSE_TEXTS = [None,] * 9
POSE_TEXTS[C_RIGHT_ARM_UP_OPEN] 			= myfont.render('RIGHT_ARM_UP_OPEN', False, C_RED, None)
POSE_TEXTS[C_RIGHT_ARM_UP_CLOSED] 			= myfont.render('RIGHT_ARM_UP_CLOSED', False, C_RED, None)
POSE_TEXTS[C_RIGHT_HAND_ON_LEFT_EAR] 		= myfont.render('RIGHT_HAND_ON_LEFT_EAR', False, C_RED, None)

POSE_TEXTS[C_LEFT_ARM_UP_OPEN] 				= myfont.render('LEFT_ARM_UP_OPEN', False, C_RED, None)
POSE_TEXTS[C_LEFT_ARM_UP_CLOSED] 			= myfont.render('LEFT_ARM_UP_CLOSED', False, C_RED, None)
POSE_TEXTS[C_LEFT_HAND_ON_RIGHT_EAR] 		= myfont.render('C_LEFT_HAND_ON_RIGHT_EAR', False, C_RED, None)

POSE_TEXTS[C_HANDS_ON_EARS] 				= myfont.render('HANDS_ON_EARS', False, C_RED, None)
POSE_TEXTS[C_CLOSE_HANDS_UP] 				= myfont.render('CLOSE_HANDS_UP', False, C_RED, None)
POSE_TEXTS[C_HANDS_ON_NECK] 				= myfont.render('HANDS_ON_NECK', False, C_RED, None)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 10,
					  qualityLevel = 0.2,
					  minDistance = 3,
					  blockSize = 5)
					  
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (5, 5),
				  maxLevel = 10,
				  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


C_NTRACK = 2
C_FEATURE_THRESHOLD = 5
C_FEATURE_THRESHOLD2 = 8
