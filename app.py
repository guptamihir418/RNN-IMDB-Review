import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb


## Pre process the text:
# User input = Movie was great!


word_index = imdb.get_word_index()
word_index_reverse = {value : key for key, value in word_index.items()}


model = tf.keras.models.load_model('Simple_rnn_imdb.h5')



def preprocess_text(user_input):
    words = user_input.lower().split()
    encoded_review = [word_index.get(word, 2) for word in words]   # if the word is not found the default value of 2 is given
    if len(encoded_review) < 10:
        encoded_review = encoded_review * 10

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

# App layout
def main():
    st.title("🎬 IMDB Movie Review Sentiment Analyzer")
    st.markdown("Welcome! Type a movie review below to see if it's positive or negative 👇")
    st.markdown("---")

    review = st.text_area("✍️ Enter your review here:")

    if st.button("🔍 Predict Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a valid review.")
        else:
            sentiment, score = predict_sentiment(review)

            st.markdown("### 🎯 Result")
            if sentiment == 'Positive':
                st.success("🌟 Sentiment: Positive")
            else:
                st.error("💔 Sentiment: Negative")

            st.markdown("### 📊 Confidence Score")
            st.progress(int(score * 100))
            st.write(f"Confidence: `{score:.2f}`")

main()