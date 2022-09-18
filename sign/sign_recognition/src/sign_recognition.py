#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skimage import exposure
from skimage import feature
from imutils import paths
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import  cv2
import rospy
import numpy as np
import argparse


##########################################
## cam conneting and open

bridge = CvBridge()
img = np.empty(shape=[0])

def img_callback(data):
    global img
    img = bridge.imgmsg_to_cv2(data, "bgr8")

rospy.init_node('sign_recognition', anonymous=True)
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

##########################################
## sign recognition code

HSV_BLUE_LOWER = np.array([80, 160, 65])
HSV_BLUE_UPPER = np.array([140, 255, 255])

cnt = np.array([0])
A1 = 0
A2 = 0
A3 = 0
B1 = 0
B2 = 0
B3 = 0

# initialize the data matrix and labels
print ("[INFO] extracting features...")
data = []
labels = []

# loop over the image paths in the training set
for imagePath in paths.list_images("/home/goeun/catkin_ws/src/sign_recognition/src/sign"):
	# extract the make of the cam
	make = imagePath.split("/")[-2]

	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# gray = cv2.resize(gray, (400, 400))

	# kernel = np.ones((1, 1), np.uint8)
	# erosion = cv2.erode(gray, kernel, iterations=1)

	# edged = imutils.auto_canny(gray)


	# find contours in the edge map, keeping only the largest one which
	# is presmumed to be the car logo
	# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# c = max(cnts, key=cv2.contourArea)
	#
	# # extract the logo of the car and resize it to a canonical width
	# # and height
	# (x, y, w, h) = cv2.boundingRect(c)
	# logo = gray[y:y + h, x:x + w]
	logo = cv2.resize(gray, (128, 128))
	# cv2.imshow("logo", logo)

	# extract Histogram of Oriented Gradients from the logo
	H = feature.hog(logo, orientations=8, pixels_per_cell=(12, 12),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

	# update the data and labels
	data.append(H)
	labels.append(make)

# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
print(model)
print("[INFO] evaluating...")

##########################################
## cam capture display

while True:
	if img.size != (640*480*3):
		continue
#	cv2.imshow("original", img)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	binary = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
	contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		area = cv2.contourArea(cnt)
		binary = cv2.drawContours(binary, [cnt], -1, (255,255,255), -1)
     
	goodContours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	kernel = np.ones((3, 3), np.uint8)
	gray = cv2.bitwise_and(binary, gray)

	for cnt in goodContours:
		area = cv2.contourArea(cnt)
		if area > 2000.0 :
			x, y, w, h = cv2.boundingRect(cnt)
			rate = w / h
			if rate > 0.6 and rate < 1.4 :
				cv2.rectangle(img, (x, y), (x+w, y+h), (200, 152, 50), 2)
				inputImage = gray[y:y+h, x:x+w]
				#
				# kernel = np.ones((1, 1), np.uint8)
				# erosion = cv2.erode(inputImage, kernel, iterations=1)
				logo = cv2.resize(inputImage, (128, 128))
				cv2.imshow("logo", logo)
				(H, hogImage) = feature.hog(logo, orientations=8, pixels_per_cell=(12, 12), \
					cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", visualise=True)

				cv2.imshow("hog", hogImage)
				pred = model.predict(H.reshape(1, -1))[0]

				cv2.putText(img, pred.title(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, \
					(70, 255, 0), 2)
                    
				cnt[0] += 1

				if pred.title() == "A1":
					A1 += 1
				if pred.title() == "A2":
					A2 += 1
				if pred.title() == "A3":
					A3 += 1
				if pred.title() == "B1":
					B1 += 1
				if pred.title() == "B2":
					B2 += 1
				if pred.title() == "B3":
					B3 += 1

	cv2.imshow("candidates", img)

	if cv2.waitKey(1) == 27:
          break

cv2.destroyAllWindows()
