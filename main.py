import json
import os
import uvicorn

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model('model.h5')

with open('texts.json') as texts_file:
    texts = json.load(texts_file)

with open('class_info.json') as class_file:
    class_info = json.load(class_file)

VOCAB_SIZE = 10000
OOV_TOK = '<OOV>'
MAX_LENGTH = 1000
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(texts)


class RequestText(BaseModel):
    text: str

@app.get("/")
def index():
    return "Ini adalah API dari Model ML TREC HEHE!"

@app.post("/predict_text")
def predict_text(req: RequestText):
    try:
        # Assuming the 'text' field is provided in the JSON payload
        text = req.text

        # Tokenize and pad the input sequence
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

        # Make a prediction using the model
        prediction = np.argmax(model.predict(padded_sequence, verbose=0))

        # Map the prediction to a class label
        result = {'prediction': class_info[str(prediction)]}

        return result
    except Exception as e:
        # Handle exceptions, e.g., missing 'text' field or other errors
        raise HTTPException(status_code=400, detail=f'Error processing request: {str(e)}')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    print(f"Listening to http://127.0.0.1:{port}")
    uvicorn.run(app, host='127.0.0.1', port=port)
