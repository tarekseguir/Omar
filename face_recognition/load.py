import cv2
import face_recognition
import os


def face_image_load():
    folder="./train/"
    images = []
    for filename in os.listdir(folder):
        path=os.path.join(folder,filename)
        img_load = face_recognition.load_image_file(path)
        img_load_encoding = face_recognition.face_encodings(img_load)[0]
        if img_load_encoding is not None:
            images.append(img_load_encoding)
    return images
    
    
def face_name():
    folder="./train/"
    names=[]
    for filename in os.listdir(folder):
        if (filename.endswith('.JPG')) or (filename.endswith('.jpg')) or (filename.endswith('.png')) or (filename.endswith('.PNG')) :
            name,_=filename.split('.')
            names.append(name)
    return names
