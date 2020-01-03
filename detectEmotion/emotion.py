#!/usr/bin/env python
# coding: utf-8


import dlib
import imutils
from imutils.video import VideoStream
import cv2
import numpy as np
import emotion_find as ef



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture("http://192.168.1.6:8080/video?x.mjpeg")
while True :
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500,height=500)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    detections = detector(gray,0)
    for detection in detections:
        emotion,X,Y = ef.emotion_find(detection,gray)
        cv2.putText(frame, emotion, (X[0],X[1]-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame,X,Y,(0,0,255),2)

    cv2.imshow("emo", frame)

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()

