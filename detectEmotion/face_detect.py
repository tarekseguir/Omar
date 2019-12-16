##this is a version of face detect it must be implimented by the EMotion code but it become slower
import cv2
import numpy as np
import dlib

def face_detector(img):
    w=25
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat_2")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    for face in faces:
        landmarks = predictor(gray, face)
        x1=landmarks.part(0).x-15
        y1=landmarks.part(0).y-50
        x2=landmarks.part(10).x+w
        y2=landmarks.part(10).y+w
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        roi_gray = gray[y1:y2, x1:x2]
        a=x1
        b=x2-x1
        c=y1
        d=y2-y1
    """
    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
    except:
        return (a,b,c,d), np.zeros((48,48), np.uint8), img
    """
    return (a,b,c,d), roi_gray, img
        
"""
        print(roi_gray)
    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
    except:
        return (x1,x2-x1,y1,y2-y1), np.zeros((48,48), np.uint8), img
    
    return (x1,x2-x1,y1,y2-y1), roi_gray, img

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    
    cv2.imshow('All', image)
    if cv2.waitKey(1) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()  
"""
