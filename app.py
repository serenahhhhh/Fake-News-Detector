from flask import Flask, request, render_template
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load tokenizer and model
try:
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = load_model('lstm_model.h5')
except Exception as e:
    print("Error loading model or tokenizer:", e)
    traceback.print_exc()

MAX_LEN = 300

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.form.get('article')

        if not input_text:
            return render_template('index.html', prediction_text="No text provided")

        # Preprocess
        seq = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Predict
        prediction = model.predict(padded)[0][0]
        label = "FAKE" if prediction < 0.5 else "REAL"
        confidence = round(float(prediction) * 100, 2)

        result = f"Prediction: {label} (Confidence: {confidence}%)"
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        traceback.print_exc()
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
