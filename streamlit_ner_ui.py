import streamlit as st
import torch
import pickle
from sklearn_crfsuite import CRF
from sklearn.metrics import f1_score


# ----------------------------
# Define the LSTM Model Class
# ----------------------------
class LSTMTagger(torch.nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentences):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out)
        return torch.argmax(logits, dim=-1)


# ----------------------------
# Load Models and Mappings
# ----------------------------
@st.cache_resource
def load_models():
    # Load the LSTM model and mappings
    with open("lstm_ner_model.pkl", "rb") as f:
        lstm_data = pickle.load(f)

    lstm_model = LSTMTagger(
        vocab_size=len(lstm_data["word2idx"]),
        tagset_size=len(lstm_data["tag2idx"]),
        embedding_dim=lstm_data["embedding_dim"],
        hidden_dim=lstm_data["hidden_dim"]
    )
    lstm_model.load_state_dict(lstm_data["model_state_dict"])
    lstm_model.eval()

    word2idx = lstm_data["word2idx"]
    tag2idx = lstm_data["tag2idx"]
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    max_len = lstm_data["max_len"]

    # Load the CRF model
    try:
        with open("crf_model.pkl", "rb") as f:
            crf_model = pickle.load(f)
    except FileNotFoundError:
        crf_model = None
        st.warning("CRF model not found. Only LSTM predictions will be available.")

    return lstm_model, crf_model, word2idx, tag2idx, idx2tag, max_len


# Load models
lstm_model, crf_model, word2idx, tag2idx, idx2tag, MAX_LEN = load_models()


# ----------------------------
# LSTM Prediction Function
# ----------------------------
def predict_lstm(sentence):
    tokens = sentence.split()
    features = [word2idx.get(word, word2idx["<UNK>"]) for word in tokens]
    features_padded = features + [word2idx["<PAD>"]] * (MAX_LEN - len(features))  # Pad to MAX_LEN
    input_tensor = torch.tensor([features_padded], dtype=torch.long)

    with torch.no_grad():
        predicted_sequence = lstm_model(input_tensor)
        predicted_tags = [idx2tag[idx.item()] for idx in predicted_sequence[0][:len(tokens)]]

    return [(token, tag) for token, tag in zip(tokens, predicted_tags)]


# ----------------------------
# CRF Prediction Function
# ----------------------------
def predict_crf(sentence):
    if crf_model is None:
        return [("N/A", "CRF Model Not Loaded")]

    tokens = sentence.split()

    def extract_features(tokens, idx):
        word = tokens[idx]
        features = {
            'word': word,
            'is_upper': word.isupper(),
            'is_title': word.istitle(),
            'is_digit': word.isdigit(),
            'word_len': len(word),
        }

        if idx > 0:
            features['prev_word'] = tokens[idx - 1]
        else:
            features['prev_word'] = "<START>"

        if idx < len(tokens) - 1:
            features['next_word'] = tokens[idx + 1]
        else:
            features['next_word'] = "<END>"

        return features

    features = [extract_features(tokens, idx) for idx in range(len(tokens))]
    predicted_tags = crf_model.predict([features])[0]
    return [(token, tag) for token, tag in zip(tokens, predicted_tags)]


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("NER Testing UI")
st.write("Test Named Entity Recognition (NER) with LSTM and CRF Models")

# User Input Sentence
sentence = st.text_input("Enter a sentence to test NER models:")

# ----------------------------
# Run Predictions
# ----------------------------
if st.button("Run NER Models"):
    if sentence.strip():
        # LSTM Predictions
        lstm_predictions = predict_lstm(sentence)
        st.write("### 📊 LSTM Model Predictions")
        lstm_output = " | ".join([f"{token}/{tag}" for token, tag in lstm_predictions])
        st.write(lstm_output)

        # CRF Predictions (if available)
        if crf_model:
            crf_predictions = predict_crf(sentence)
            st.write("### 📊 CRF Model Predictions")
            crf_output = " | ".join([f"{token}/{tag}" for token, tag in crf_predictions])
            st.write(crf_output)
        else:
            st.warning("CRF model is not available. Only LSTM predictions are displayed.")
    else:
        st.warning("Please enter a sentence to test.")

# ----------------------------
# Footer
# ----------------------------
st.write("---")
st.write("Developed with ❤️ using Streamlit, PyTorch, and sklearn-crfsuite.")
