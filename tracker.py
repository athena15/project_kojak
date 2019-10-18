import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Select Region of Interest (ROI)
r = cv2.selectROI(frame)

# Crop image
imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

# setup initial location of window
r, h, c, w = 250, 400, 400, 400
track_window = (c, r, w, h)
# set up the ROI for tracking
roi = imCrop
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                   np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while (1):
        ret, frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.imshow('img2', img2)
            k = cv2.waitKey(60) & 0xff
            if k == 27:  # if ESC key
                break
            else:
                cv2.imwrite(chr(k) + ".jpg", img2)
        else:
            break


