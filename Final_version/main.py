from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model

def eye_brow_distance(leye,reye):
    global points
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    return distq




def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)



def emotion_finder(faces,frame):
    global emotion_classifier
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    emotion = EMOTIONS[preds.argmax()]
    #print(label)
    if label in ['scared','sad']:
        label = 'stressed'
    else:
        label = 'not stressed'
    return emotion , label
    
def normalize_values(points,disp):
    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    stress_value = np.exp(-(normalized_value))
    #print(stress_value)
    val=stress_value*100
    level=""
    color=(0,255,0)
    if val>=75:
        level="High Stress"
        color=(0,0,255)
    elif val>=50 and val < 75:
        level="low_stress"
        color=(0,255,255)
    
    return stress_value,level,color
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_classifier = load_model("emotion_model.hdf5", compile=False)
cap = cap = cv2.VideoCapture(0)
points = []
while True :
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500,height=500)
    
    
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    detections = detector(gray)


    for detection in detections:
        landmarks = predictor(gray, detection)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        landmarks = predictor(gray, detection)
        theRealEmotion , emotion = emotion_finder(detection,gray)
        cv2.putText(frame, theRealEmotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        shape = predictor(frame,detection)
        shape = face_utils.shape_to_np(shape)
           
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]
            
        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)

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


        #cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

        distq = eye_brow_distance(leyebrow[-1],reyebrow[0])
        stress_value,stress_label,color = normalize_values(points,distq)
        cv2.putText(frame,"stress level:{}".format(str(int(stress_value*100))),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame,stress_label,(60,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    cv2.imshow("Frame", frame)

    #key = cv2.waitKey(1) & 0xFF
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
cap.release()
"""
plt.plot(range(len(points)),points,'ro')
plt.title("Stress Levels")
plt.show()
"""
