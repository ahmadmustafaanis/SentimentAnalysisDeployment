import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from model import preProcess_data
from pydantic import BaseModel
import tensorflow as tf
import re
def preProcess_data(text):
    
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

app = FastAPI()

data = pd.read_csv('archive/Sentiment.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)



def my_pipeline(text):
  text_new = preProcess_data(text)
  X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
  X = pad_sequences(X, maxlen=28)
  return X


class inputToModel(BaseModel):
    text:str


@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}



@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''<form method="post"> 
    <input type="text" maxlength="28" name="text" value="Text Emotion to be tested"/>  
    <input type="submit"/> 
    </form>'''



@app.post('/predict')
def predict(text:str = Form(...)):
    clean_text = my_pipeline(text)
    loaded_model = tf.keras.models.load_model('sentiment.h5')
    predictions = loaded_model.predict(clean_text)
    sentiment = int(np.argmax(predictions))
    probability = max(predictions.tolist()[0])
    print(sentiment)
    if sentiment==0:
        t_sentiment = 'negative'
    elif sentiment==1:
        t_sentiment = 'neutral'
    elif sentiment==2:
        t_sentiment='postive'
    
    return {
        "ACTUALL SENTENCE": text,
        "PREDICTED SENTIMENT": t_sentiment,
        "Probability": probability
    }