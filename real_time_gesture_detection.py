import cv2
import numpy as np
import copy
import math
import datetime

from keras.preprocessing import image as image_utils

prediction = str(0)

def predict_rgb_image(img):
	result = gesture_names[model.predict_classes(img)[0]]
	print(result)
	return (result)

from keras.models import load_model
model = load_model('/Users/brenner/project_kojak/models/new_new_new.h5')
# model = load_model('/Users/brenner/project_kojak/models/vgg_new_model.h5')
# model = load_model('/Users/brenner/project_kojak/drawing_VGG.h5')
# model = load_model('/Users/brenner/project_kojak/models/best_model')

# gesture_names = {0: 'C',`
# 				 1: 'Fist',
# 				 2: 'L',
# 				 3: 'Okay',
# 				 4: 'Palm',
# 				 5: 'Peace'}

gesture_names = {0: 'Fist',
				 1: 'L',
				 2: 'Okay',
				 3: 'Palm',
				 4: 'Peace'}

def predict_rgb_image_vgg(img):
	# img2rgb = image_utils.load_img(path=path, target_size=(224, 224))
	# img2rgb = image_utils.img_to_array(img2rgb)
#     image_rgb.append(img2rgb)
#     img2rgb = img2rgb.reshape(1, 224, 224, 3)
	img = np.array(img, dtype='float32')
	img /= 255
	pred_array = model.predict(img)
	print(f'pred_array: {pred_array}')
	result = gesture_names[np.argmax(pred_array)]
	print(f'Result: {result}')
	print(max(pred_array[0]))
	print(result)
	return result

###STOP

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
img_counter = 170

# variableslt
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
	print("! Changed threshold to "+str(thr))


def removeBG(frame):
	fgmask = bgModel.apply(frame,learningRate=learningRate)
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	# res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

	kernel = np.ones((3, 3), np.uint8)
	fgmask = cv2.erode(fgmask, kernel, iterations=1)
	res = cv2.bitwise_and(frame, frame, mask=fgmask)
	return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
	#  convexity defect
	hull = cv2.convexHull(res, returnPoints=False)
	if len(hull) > 3:
		defects = cv2.convexityDefects(res, hull)
		if type(defects) != type(None):  # avoid crashing.   (BUG not found)

			cnt = 0
			for i in range(defects.shape[0]):  # calculate the angle
				s, e, f, d = defects[i][0]
				start = tuple(res[s][0])
				end = tuple(res[e][0])
				far = tuple(res[f][0])
				a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
				b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
				c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
				angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
				if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
					cnt += 1
					cv2.circle(drawing, far, 8, [211, 84, 0], -1)
			return True, cnt
	return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


while camera.isOpened():
	ret, frame = camera.read()
	threshold = cv2.getTrackbarPos('trh1', 'trackbar')
	frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
	frame = cv2.flip(frame, 1)  # flip the frame horizontally
	cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
				 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
	cv2.putText(frame, f"Prediction: {prediction}" , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))  # Draw the text
	cv2.imshow('original', frame)

	#  Main operation
	if isBgCaptured == 1:  # this part wont run until background captured
		img = removeBG(frame)
		img = img[0:int(cap_region_y_end * frame.shape[0]),
					int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
		cv2.imshow('mask', img)

		# convert the image into binary image
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
		# cv2.imshow('blur', blur)
		# ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
		ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Remove cv2.THRESH_OTSU if needed

		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY
		cv2.imshow('ori', thresh)


		# get the contours
		thresh1 = copy.deepcopy(thresh)
		_,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		length = len(contours)
		maxArea = -1
		if length > 0:
			for i in range(length):  # find the biggest contour (according to area)
				temp = contours[i]
				area = cv2.contourArea(temp)
				if area > maxArea:
					maxArea = area
					ci = i

			res = contours[ci]
			hull = cv2.convexHull(res)
			drawing = np.zeros(img.shape, np.uint8)
			cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
			cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

			isFinishCal,cnt = calculateFingers(res,drawing)
			if triggerSwitch is True:
				if isFinishCal is True and cnt <= 2:
					print (cnt)
					#app('System Events').keystroke(' ')  # simulate pressing blank space


		cv2.imshow('output', drawing)

	# Keyboard OP
	k = cv2.waitKey(10)
	if k == 27:  # press ESC to exit
		break
	elif k == ord('b'):  # press 'b' to capture the background
		bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
		isBgCaptured = 1
		print( '!!!Background Captured!!!')
	elif k == ord('r'):  # press 'r' to reset the background
		bgModel = None
		triggerSwitch = False
		isBgCaptured = 0
		print ('!!!Reset BackGround!!!')
	elif k == 32:
		# SPACE pressed

		# img_name = f"/Users/brenner/project_kojak/frames/drawings/drawing_peace_{img_counter}.jpg".format(img_counter)
		# cv2.imwrite(img_name, drawing)
		# print("{} written!".format(img_name))
		# img_counter +=1
		# # print(type(thresh))


		drawing = cv2.resize(drawing, (224, 224))
		drawing = drawing.reshape(1, 224, 224, 3)
		prediction = predict_rgb_image_vgg(drawing)



		# if prediction:
		# 	cv2.putText(frame, 'detected:', (50, 50), self.font, 0.8, (0, 0, 0), 2)

	# img_name2 = f"/Users/brenner/project_kojak/misc/drawing_palm_{img_counter}.jpg".format(img_counter)
		# cv2.imwrite(img_name2, drawing)
		# print("{} written!".format(img_name2))
	elif k == ord('t'):
		print('Tracker turned on.')



		cap = cv2.VideoCapture(0)
		# take first frame of the video
		ret, frame = cap.read()

		# Read image

		# Select ROI
		r = cv2.selectROI(frame)

		# Crop image
		imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

		# setup initial location of window
		r, h, c, w = 250, 400, 400, 400  # simply hardcoded the values
		track_window = (c, r, w, h)
		# set up the ROI for tracking
		roi = imCrop
		hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
		roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
		cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
		# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
		term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
		while (1):
			ret, frame = cap.read()
			if ret == True:
				# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				# dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
				# # apply meanshift to get the new location
				# ret, track_window = cv2.meanShift(dst, track_window, term_crit)
				# # Draw it on image
				# x, y, w, h = track_window
				# img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
				# cv2.imshow('img2', img2)
				# k = cv2.waitKey(60) & 0xff
				# if k == 27:
				# 	break
				# else:
				# 	cv2.imwrite(chr(k) + ".jpg", img2)
				hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
				# apply meanshift to get the new location
				ret, track_window = cv2.CamShift(dst, track_window, term_crit)
				# Draw it on image
				pts = cv2.boxPoints(ret)
				pts = np.int0(pts)
				img2 = cv2.polylines(frame, [pts], True, (0,255,0), 2)
				cv2.imshow('img2', img2)
				k = cv2.waitKey(60) & 0xff
				if k == 27:
					break
				else:
					cv2.imwrite(chr(k) + ".jpg", img2)

			else:
				break
		cv2.destroyAllWindows()
		cap.release()

	elif k == ord('n'):
		triggerSwitch = True
		print ('!!!Trigger On!!!')