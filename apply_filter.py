import time
import cv2
import numpy as np
import glob
import os
import dlib
import argparse

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

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

def main(file, output, frame_rate=30):
	if (file == "camera"):
		video_capture = cv2.VideoCapture(0)
	else:
		video_capture = cv2.VideoCapture(file)
	ret, frame = video_capture.read()
	if (output != None):
		out = cv2.VideoWriter(output,fourcc, frame_rate, (frame.shape[1], frame.shape[0]))

	while ret:
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
		if (output != None):
			out.write(frame)
		else:
			cv2.imshow("", frame)
		ret, frame = video_capture.read()
		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break
		# Release handle to the webcam
	if (output != None):
		out.release()
	video_capture.release()
	cv2.destroyAllWindows()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", type=str, help="give video file for filter write camera if you want to use webcam", required=True)
	parser.add_argument("-o", "--output", type=str, help="give output name for video in .mp4 format")
	parser.add_argument("-fr", "--frame_rate", type=str, help="give video frame", default=30)
	args = parser.parse_args()

	file = args.file
	output = args.output
	frame_rate = args.frame_rate
	main(file, output, frame_rate)
