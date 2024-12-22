import streamlit as st
import torch
import joblib
import pickle
from sklearn_crfsuite import CRF

# ----------------------------
# Define Model Classes
# ----------------------------

# BiLSTM Model Class
class BiLSTMTagger(torch.nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTMTagger, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentences):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out)
        return logits


# LSTM Model Class
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
def load_bilstm_model():
    saved_data = joblib.load("bilstm_ner_model.pkl")
    model = BiLSTMTagger(
        len(saved_data['word2idx']),
        len(saved_data['tag2idx']),
        saved_data['embedding_dim'],
        saved_data['hidden_dim']
    )
    model.load_state_dict(saved_data['model_state_dict'])
    model.eval()
    return model, saved_data['word2idx'], saved_data['idx2tag'], saved_data.get('max_len', 50)


@st.cache_resource
def load_lstm_model():
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
    return lstm_model, lstm_data["word2idx"], lstm_data["idx2tag"], lstm_data["max_len"]


@st.cache_resource
def load_crf_model():
    """
    Load the CRF model from a pickle file.
    """
    try:
        with open("crf_ner_model.pkl", "rb") as f:
            crf_model = pickle.load(f)
        return crf_model
    except FileNotFoundError:
        st.error("CRF model file not found. Ensure 'crf_model.pkl' exists in your directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the CRF model: {e}")
        return None


# Load all models
bilstm_model, bilstm_word2idx, bilstm_idx2tag, BILSTM_MAX_LEN = load_bilstm_model()
lstm_model, lstm_word2idx, lstm_idx2tag, LSTM_MAX_LEN = load_lstm_model()
crf_model = load_crf_model()


# ----------------------------
# Preprocessing
# ----------------------------

def preprocess_sentence(sentence, word2idx, max_len):
    sentence_idx = [word2idx.get(word, word2idx.get("<UNK>", 0)) for word in sentence]
    sentence_idx = sentence_idx[:max_len] + [word2idx.get("<PAD>", 0)] * (max_len - len(sentence_idx))
    return torch.tensor([sentence_idx], dtype=torch.long)


# ----------------------------
# Prediction Functions
# ----------------------------

def predict_bilstm(sentence):
    tokens = sentence.split()
    input_tensor = preprocess_sentence(tokens, bilstm_word2idx, BILSTM_MAX_LEN)
    with torch.no_grad():
        outputs = bilstm_model(input_tensor)
        predicted_indices = torch.argmax(outputs, dim=-1).cpu().numpy()[0]
    predicted_tags = [bilstm_idx2tag.get(idx, "O") for idx in predicted_indices[:len(tokens)]]
    return list(zip(tokens, predicted_tags))


def predict_lstm(sentence):
    tokens = sentence.split()
    features = [lstm_word2idx.get(word, lstm_word2idx["<UNK>"]) for word in tokens]
    features_padded = features + [lstm_word2idx["<PAD>"]] * (LSTM_MAX_LEN - len(features))
    input_tensor = torch.tensor([features_padded], dtype=torch.long)
    with torch.no_grad():
        predicted_sequence = lstm_model(input_tensor)
        predicted_tags = [lstm_idx2tag[idx.item()] for idx in predicted_sequence[0][:len(tokens)]]
    return list(zip(tokens, predicted_tags))


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
    if idx < len(tokens) - 1:
        features['next_word'] = tokens[idx + 1]
    return features


def predict_crf(sentence):
    if crf_model is None:
        return [("N/A", "CRF Model Not Loaded")]
    tokens = sentence.split()
    features = [extract_features(tokens, idx) for idx in range(len(tokens))]
    predicted_tags = crf_model.predict([features])[0]
    return list(zip(tokens, predicted_tags))


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("NER Testing UI with BiLSTM, LSTM, and CRF Models")
st.write("Test Named Entity Recognition (NER) with BiLSTM, LSTM, and CRF Models.")

# Model Selection
model_choice = st.selectbox("Choose a Model:", ["BiLSTM", "LSTM", "CRF"])

# User Input Sentence
sentence = st.text_area("Enter a sentence for NER:")

if st.button("Run NER Model"):
    if sentence.strip():
        if model_choice == "BiLSTM":
            predictions = predict_bilstm(sentence)
        elif model_choice == "LSTM":
            predictions = predict_lstm(sentence)
        elif model_choice == "CRF":
            predictions = predict_crf(sentence)

        st.write(f"### 📊 {model_choice} Model Predictions")
        output = " | ".join(
            [f"**{token}/{tag}**" if tag != "O" else f"{token}/{tag}" for token, tag in predictions]
        )
        st.markdown(output)
    else:
        st.warning("Please enter a sentence to run predictions.")

# Footer
st.write("---")
st.write("Developed with ❤️ using Streamlit, PyTorch, and sklearn-crfsuite.")
