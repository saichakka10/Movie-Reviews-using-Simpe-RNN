# import libraries 
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load the model
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

# Load the pretained model with relu activation function
model=load_model('simple_rnn_imdb.h5')


# Function to decode reviews
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,"?") for i in encoded_review])


# Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


# Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0]>0.5 else "Negative"
    return sentiment, prediction[0][0]


import streamlit as st
st.title("TMDB Movie Review-Sentiment-Analysis")
st.write("Enter a movie review to classify it as Positive or Negative")

user_input = st.text_area("Movie Review")

if st.button('classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0]>0.5 else "Negative"
    # Display the result
    st.write(f"Sentiment:{sentiment}")
    st.write(f"Prediction Score:{prediction[0][0]}")
else:
    st.write("Please enter a movie review")