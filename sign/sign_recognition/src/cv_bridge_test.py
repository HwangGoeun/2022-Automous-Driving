#!/usr/bin/env python
# -*- coding: utf-8 -*-

import  cv2
import rospy
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
img = np.empty(shape=[0])

def img_callback(data):
    global img
    img = bridge.imgmsg_to_cv2(data, "bgr8")

rospy.init_node('cam_tune', anonymous=True)
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

HSV_RED_LOWER = np.array([0, 100, 100])
HSV_RED_UPPER = np.array([10, 255, 255])
HSV_RED_LOWER1 = np.array([160, 100, 100])
HSV_RED_UPPER1 = np.array([179, 255, 255])

HSV_YELLOW_LOWER = np.array([10, 80, 120])
HSV_YELLOW_UPPER = np.array([40, 255, 255])

HSV_BLUE_LOWER = np.array([80, 160, 65])
HSV_BLUE_UPPER = np.array([140, 255, 255])

while not rospy.is_shutdown():
    if img.size != (640*480*3):
        continue
    cv2.imshow("display", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    redBinary = cv2.inRange(hsv, HSV_RED_LOWER, HSV_RED_UPPER)
    redBinary1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
    redBinary = cv2.bitwise_or(redBinary, redBinary1)
	# cv2.imshow("red", redBinary)
    yellowBinary = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
	# cv2.imshow("yellow", yellowBinary)
    blueBinary = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    bgrblue = cv2.inRange(img, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    binary = cv2.bitwise_or( blueBinary, (redBinary))

    cv2.imshow("gray", gray)
    cv2.imshow("hsv", hsv)
    cv2.imshow("binary", binary)
    cv2.imshow("blueBinary", blueBinary)
    cv2.imshow("bgrBlueBinary", bgrblue)

    cv2.waitKey(33)

