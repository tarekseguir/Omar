import cv2
import speech_recognition as sr
import threading
import numpy as np
import matplotlib.pyplot as plt
"""
nadia = cv2.imread("Nadia_Murad.jpg",0)
denis = cv2.imread("Denis_Mukwege.jpg",0)
solvay = cv2.imread("solvay_conference.jpg",0)
cv2.imshow("klj",solvay)
"""
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

def detect_face(img):
    face_img=img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)

    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img

def adj_detect_face(img):
    face_img=img.copy()
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)#find balance between this two parametre to get a good detection (toomuch faces or the minumim

    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img



def detect_eye(img):
    eye_img=img.copy()
    eye_rects = eye_cascade.detectMultiScale(eye_img)

    for(x,y,w,h) in eye_rects:
        cv2.rectangle(eye_img,(x,y),(x+w,y+h),(255,255,255),10)
    return eye_img
"""
result = detect_eye(denis)#cet photo est indetectable(yeux) car la partie blanc dans l'ouiel est presque ''noir''(invisible)
cv2.imshow('eye',result)
result2 = detect_eye(nadia)
cv2.imshow('eye2',result2)
result12 = detect_eye(solvay)
cv2.imshow('eye12',result12)
"""
#Cas d'un Video
def face_detection():

    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read(0)
        frame=detect_face(frame)
        cv2.imshow('mlk',frame)
        if cv2.waitKey(33) == ord('a') :
            break
    cap.release()
    cv2.destroyAllWindows()
def speech_detection():
    r = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("phase 1:")
            audio = r.listen(source)

            try:
                text = r.recognize_google(audio)
                print(text)
            except:
                print("NAN")


speech =  threading.Thread(target=speech_detection)
face = threading.Thread(target=face_detection)
speech.start()
face.start()
