import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat_2")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)



        # face detection ----------------------------------------------------------------------------
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        landmarks = predictor(gray, face)
        #--------------------------------------------------------------------------------------------




        # The right eye detection :------------------------------------------------------------------
        left_point_r = (landmarks.part(36).x, landmarks.part(36).y)
        right_point_r = (landmarks.part(39).x, landmarks.part(39).y)
        center_top_r = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom_r = midpoint(landmarks.part(41), landmarks.part(40))
        hor_line_r = cv2.line(frame, left_point_r, right_point_r, (0, 255, 0), 2)
        ver_line_r = cv2.line(frame, center_top_r, center_bottom_r, (0, 255, 0), 2)
        #--------------------------------------------------------------------------------------------


        # The left eye detection :-------------------------------------------------------------------
        left_point_l = (landmarks.part(42).x, landmarks.part(42).y)
        right_point_l = (landmarks.part(45).x, landmarks.part(45).y)
        center_top_l = midpoint(landmarks.part(43), landmarks.part(44))
        center_bottom_l = midpoint(landmarks.part(47), landmarks.part(46))
        hor_line_l = cv2.line(frame, left_point_l, right_point_l, (0, 255, 0), 2)
        ver_line_l = cv2.line(frame, center_top_l, center_bottom_l, (0, 255, 0), 2)
        #-------------------------------------------------------------------------------------------

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
