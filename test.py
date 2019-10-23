

import cv2


def main():

	cap = cv2.VideoCapture(0)

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



	while True:
		try:
			cap_res, cap_frame = cap.read()
			cv2.imshow('', cap_frame)
			cv2.waitKey(1)


		except:
			break


if __name__ == "__main__":
	main()
