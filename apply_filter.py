import time
import cv2
import numpy as np
import glob
import os
import dlib

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 1280)
video_capture.set(4, 720)

saved_face_encodings = []
names = []

imgMustache = cv2.imread("mustache.png", -1)
orig_mask = imgMustache[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

imgGlass = cv2.imread("glasses.png", -1)
orig_mask_g = imgGlass[:,:,3]
orig_mask_inv_g = cv2.bitwise_not(orig_mask_g)
imgGlass = imgGlass[:,:,0:3]
origGlassHeight, origGlassWidth = imgGlass.shape[:2]


predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

while True:
	ret, frame = video_capture.read()
	
	dets = cnn_face_detector(frame, 1)
	for k, d in enumerate(dets):
		shape = predictor(frame, d.rect)
        
		mustacheWidth = abs(3 * (shape.part(31).x - shape.part(35).x))
		mustacheHeight = int(mustacheWidth * origMustacheHeight / origMustacheWidth) - 10
		mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
		mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
		mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
		y1 = int(shape.part(33).y - (mustacheHeight/2)) + 10
		y2 = int(y1 + mustacheHeight)
		x1 = int(shape.part(51).x - (mustacheWidth/2))
		x2 = int(x1 + mustacheWidth)
		roi = frame[y1:y2, x1:x2]
		roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
		roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
		frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)
		
		glassWidth = abs(shape.part(16).x - shape.part(1).x)
		glassHeight = int(glassWidth * origGlassHeight / origGlassWidth)
		glass = cv2.resize(imgGlass, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
		mask = cv2.resize(orig_mask_g, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
		mask_inv = cv2.resize(orig_mask_inv_g, (glassWidth,glassHeight), interpolation = cv2.INTER_AREA)
		y1 = int(shape.part(24).y)
		y2 = int(y1 + glassHeight)
		x1 = int(shape.part(27).x - (glassWidth/2))
		x2 = int(x1 + glassWidth)
		roi1 = frame[y1:y2, x1:x2]
		roi_bg = cv2.bitwise_and(roi1,roi1,mask = mask_inv)
		roi_fg = cv2.bitwise_and(glass,glass,mask = mask)
		frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)
		#'''
	cv2.imshow("", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
	# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()