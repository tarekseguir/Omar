import cv2
import numpy as np
from imutils import face_utils
from keras.preprocessing.image import img_to_array
from keras.models import load_model

emotion_classifier = load_model("emotion_model.hdf5", compile=False)

def emotion_find(faces,frame):
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
    return label , (x,y),(x+w,y+h)
