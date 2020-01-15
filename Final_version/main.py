import pandas as pd
from scipy.spatial import distance as dist
from imutils.video import VideoStream
import math
import argparse
from imutils import face_utils
import speech_recognition as sr
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model
f=open("outputs.txt","w")
data = {'gender':[],'emotion':[],'stress_level':[],'speech':[]}
df = pd.DataFrame(data)
def eye_brow_distance(leye,reye):
    global points
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    return distq


def callback(recognizer, audio):                          # this is called from the background thread
    try:
        #r.adjust_for_ambient_noise(audio,duration=0.5)
        #recognizer.adjust_for_ambient_noise(audio, duration = 1)
        recognizer.dynamic_energy_threshold = True
##        recognizer_instance.dynamic_energy_adjustment_damping = 0.15
##        This value should be between 0 and 1 When this value is 1, dynamic adjustment has no effect
##        Lower values allow for faster adjustment, but also make it more likely to miss certain phrases (especially those with slowly changing volume)
##        recognizer_instance.dynamic_energy_adjustment_ratio = 1.5
##        the default value of 1.5 means that speech is at least 1.5 times louder than ambient noise.
        recognizer.pause_threshold = 1 #Represents the minimum length of silence (in seconds) that will register as the end of a phrase. Can be changed.
        w=recognizer.recognize_google(audio,language = "en-US")
        with open("outputs.txt","a") as f:
            f.write(w+'\n')
        #tab.append(w)
        print("You said " + w)  # received audio data, now need to recognize it
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    except LookupError:
        print("Oops! Didn't catch that")
    except Exception as e:
            #callback(recognizer,audio)
            print("")

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
    return frameOpencvDnn,faceBoxes


def emotion_finder(faces,frame):
    try:
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
    except:
        print("more than one")
        label = 'not stressed'
        emotion = 'neutral'
    return emotion , label
    
def normalize_values(points,disp):
    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points)-20)
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

gender = theRealEmotion = w = None
parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
padding=20
r = sr.Recognizer()
stop_listening =r.listen_in_background(sr.Microphone(), callback)#r:recognizer #sr.Microphone:audio
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_classifier = load_model("emotion_model.hdf5", compile=False)
cap = cv2.VideoCapture(0)
points = []
i = k = j =0
while True :
    #print("hey hey")
    hasFrame,frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500,height=500)
    
    
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    detections = detector(gray)

    resultImg,faceBoxes=highlightFace(faceNet,frame)
    for detection in detections:
        landmarks = predictor(gray, detection)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        landmarks = predictor(gray, detection)
        theRealEmotion , emotion = emotion_finder(detection,gray)
        
        shape = predictor(frame,detection)
        shape = face_utils.shape_to_np(shape)
           
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]
            
        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)

        

        #cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)
        distq = eye_brow_distance(leyebrow[-1],reyebrow[0])
        stress_value,stress_label,color = normalize_values(points,distq)
        try:
            cv2.putText(frame,"stress level:{}".format(str(int(stress_value*100))),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,stress_label,(60,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i+=1
            if i>50:
                print( theRealEmotion+ " " + "stress level:{}".format(str(int(stress_value*100))) + " " + stress_label)
                i=0
        except:
            continue


    
    if not faceBoxes:
        print(" ")
    
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')
        cv2.putText(frame, theRealEmotion, (faceBox[0]+40, faceBox[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'{gender}', (faceBox[0]-20, faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow("Frame", frame)
    k+=1
    if(k>20):
        df.loc[j] = [gender,theRealEmotion,str(int(stress_value*100)),w]
        print(df)
        j+=1
        k=0
    #key = cv2.waitKey(1) & 0xFF
    if cv2.waitKey(1) & 0xFF == 27:
        break
df.to_csv('myDataFrame.csv')
cv2.destroyAllWindows()
cap.release()
"""
plt.plot(range(len(points)),points,'ro')
plt.title("Stress Levels")
plt.show()
"""
