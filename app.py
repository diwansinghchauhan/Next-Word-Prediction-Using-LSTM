import streamlit as st
import numpy as np
import pickle
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model=load_model('next_word_GRU.keras')

# Load the tokenizer
with open('tokenizer_gru.pickle','rb') as handle:
    tokenizer=pickle.load(handle)


# Function to predict the next word
def predict_next_word(model, tokenizer, train_text, max_len):
    token_list = tokenizer.texts_to_sequences([train_text])[0]
    if len(token_list) >= max_len:
        token_list = token_list[-(max_len-1):]  # Ensure the sequence length matches max_len-1
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM")
input_text=st.text_input("Enter the sequence of words")
if st.button("Predict Next Word"):
        max_len=model.input_shape[1]+1
        next_word=predict_next_word(model,tokenizer,input_text,max_len)
        st.write(f'Next word: {next_word}')