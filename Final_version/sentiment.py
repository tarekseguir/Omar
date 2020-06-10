import pandas as pd
import numpy as np
from keras.models import load_model
sentiment_model = load_model("model1.h5", compile=False)
pd.read_csv( 'myDataFrame.csv', header=None)
def sentiment_finder(x):
    try:
        global sentiment_model
        preds = sentiment_model.predict(x)
for x in df['speech']:
    
