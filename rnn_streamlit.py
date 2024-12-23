import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load resources
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('model_compatible_final.keras', compile=False)
    word2idx = {
        'I': 2, 'love': 3, 'NLP': 4, '<PAD>': 4012, '<UNK>': 4012
    }
    maxlen = 361
    return model, word2idx, maxlen


def predict_sentence(sentence, model, word2idx, maxlen):
    BIO_MAP = {0: 'O', 1: 'B', 2: 'I'}
    words = sentence.split()
    input_dim = model.layers[0].input_dim

    # Validate and prepare input
    if "<PAD>" not in word2idx or word2idx["<PAD>"] >= input_dim:
        word2idx["<PAD>"] = input_dim - 1
    if "<UNK>" not in word2idx or word2idx["<UNK>"] >= input_dim:
        word2idx["<UNK>"] = input_dim - 1

    X_test = [[min(word2idx.get(w, word2idx.get("<UNK>", 0)), input_dim - 1) for w in words]]
    X_test = pad_sequences(X_test, maxlen=maxlen, padding="post", value=word2idx["<PAD>"]).astype('int32')

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=-1)[0]
    predicted_tags = [BIO_MAP.get(tag_idx, 'O') for tag_idx in y_pred_classes[:len(words)]]

    # Heuristic Fix for BIO tags
    for i, (word, tag) in enumerate(zip(words, predicted_tags)):
        if tag == 'I' and (i == 0 or predicted_tags[i - 1] == 'O'):
            predicted_tags[i] = 'B'

    corrected_tags = list(zip(words, predicted_tags))
    return corrected_tags


# Streamlit UI
st.title("Named Entity Recognition with RNN")
sentence = st.text_input("Enter a sentence for prediction:", "I love NLP")

if st.button("Predict"):
    model, word2idx, maxlen = load_resources()
    predicted_tags = predict_sentence(sentence, model, word2idx, maxlen)

    st.write("\n### Sentence:")
    st.write(sentence)
    st.write("\n### Predicted Tags:")
    for word, tag in predicted_tags:
        st.write(f"**{word}**: {tag}")
