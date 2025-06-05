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


def generate_text(model, tokenizer, seed_text, max_len, num_words):
    output_text = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = token_list[-(max_len - 1):]  # keep only the last max_len-1 tokens
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
        
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        
        # Find the word by index
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_text += " " + word
                break
    return output_text

# streamlit app
st.title("Text Generation With LSTM")
input_text=st.text_input("Enter the sequence of words")
if st.button("Generate Text"):
        max_len=model.input_shape[1]+1
        next_word = generate_text(model, tokenizer, input_text, max_len, num_words=3)
        st.write(next_word)