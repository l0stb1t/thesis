import pygame
import cv2

C_MIDHIP = 18
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
C_LWRIST = 9
C_RWRIST = 10
C_LELBOW = 7
C_RELBOW = 8
C_LSHOULDER = 5
C_RSHOULDER = 6
C_REAR = 4
C_LEAR = 3

C_KP_NAMES = (
	'nose', 'left eye', 'right eye', 'left ear', 'right ear',
	'left shoulder', 'right shoulder',
	'left elbow', 'right elbow',
	'left wrist', 'right wrist',
	'left hip', 'right hip',
	'left knee', 'right knee',
	'left ankle', 'right ankle',
	'neck', 'mid hip'
)

C_PAIRS = ((C_LSHOULDER, C_LHIP), (C_RSHOULDER, C_RHIP), (C_LHIP, C_RHIP), (C_LKNEE, C_LHIP), (C_RKNEE, C_RHIP), (C_LANKLE, C_LKNEE), (C_RANKLE, C_RKNEE), (C_RWRIST, C_RELBOW), (C_LWRIST, C_LELBOW), (C_RELBOW, C_RSHOULDER), (C_LELBOW, C_LSHOULDER), (C_LSHOULDER, C_RSHOULDER))

C_NKP 				= 19
C_MAXPOSE 			= 10
C_KP_THRESHOLD 		= 0.05
C_PSCORE_THRESHOLD 	= 0.3

C_RED 		= (255, 0, 0)
C_GREEN 	= (0, 255, 0)
C_BLUE 		= (0, 0, 255)
C_YELLOW 	= (255, 255, 0)
C_COLORS 	= (C_RED, C_GREEN, C_BLUE)

C_RIGHT_ARM_UP_OPEN 			= 0
C_RIGHT_ARM_UP_CLOSED 			= 1
C_RIGHT_HAND_ON_LEFT_EAR 		= 2
C_LEFT_ARM_UP_OPEN 				= 3
C_LEFT_ARM_UP_CLOSED 			= 4
C_LEFT_HAND_ON_RIGHT_EAR 		= 5
C_HANDS_ON_EARS 				= 6
C_CLOSE_HANDS_UP 				= 7
C_HANDS_ON_NECK 				= 8

C_GESTURE_NAMES = [
	'right arm up open', 'right arm up closed', 'right hand on left ear',
	'left arm up open', 'left arm up closed', 'left hand on right ear',
	'hands on ear', 'close hands up', 'hands on neck'
]

pygame.font.init()
FONT = pygame.font.SysFont(None, 20)
LABELS = [None,]*10

GESTURE_NAMES = [None,] * 9
GESTURE_NAMES[C_RIGHT_ARM_UP_OPEN] 		= FONT.render('RIGHT_ARM_UP_OPEN', False, C_GREEN, None)
GESTURE_NAMES[C_RIGHT_ARM_UP_CLOSED] 	= FONT.render('RIGHT_ARM_UP_CLOSED', False, C_GREEN, None)
GESTURE_NAMES[C_RIGHT_HAND_ON_LEFT_EAR] = FONT.render('RIGHT_HAND_ON_LEFT_EAR', False, C_GREEN, None)

GESTURE_NAMES[C_LEFT_ARM_UP_OPEN] 		= FONT.render('LEFT_ARM_UP_OPEN', False, C_GREEN, None)
GESTURE_NAMES[C_LEFT_ARM_UP_CLOSED] 	= FONT.render('LEFT_ARM_UP_CLOSED', False, C_GREEN, None)
GESTURE_NAMES[C_LEFT_HAND_ON_RIGHT_EAR] = FONT.render('C_LEFT_HAND_ON_RIGHT_EAR', False, C_GREEN, None)

GESTURE_NAMES[C_HANDS_ON_EARS] 			= FONT.render('HANDS_ON_EARS', False, C_GREEN, None)
GESTURE_NAMES[C_CLOSE_HANDS_UP] 		= FONT.render('CLOSE_HANDS_UP', False, C_GREEN, None)
GESTURE_NAMES[C_HANDS_ON_NECK] 			= FONT.render('HANDS_ON_NECK', False, C_GREEN, None)

C_STANDING = FONT.render('STANDING', False, C_GREEN, None)
C_SITTING = FONT.render('SITTING', False, C_BLUE, None)
C_UNKNOW = FONT.render('UNKNOW', False, C_RED, None)

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
