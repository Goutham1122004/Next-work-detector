import streamlit as st
import torch
import string
from transformers import BertTokenizer, BertForMaskedLM

# Load the model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model(model_name):
    if model_name.lower() == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
        return tokenizer, model
    return None, None

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])

def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

def get_all_predictions(text_sentence, top_clean=5, top_k=5):
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    predictions = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return predictions

def get_prediction_eos(input_text, top_clean, top_k):
    input_text += ' <mask>'
    return get_all_predictions(input_text, top_clean, top_k)

# Streamlit app
st.title("Next Word Prediction with PyTorch")
st.sidebar.title("Settings")

top_k = st.sidebar.slider("How many words do you need", 1, 25, 1)
model_name = st.sidebar.selectbox("Select Model to Apply", options=['BERT'], index=0)

bert_tokenizer, bert_model = load_model(model_name)
input_text = st.text_area("Enter your text here", height=200)

if input_text:
    result = get_prediction_eos(input_text, top_clean=top_k, top_k=top_k)
    st.text_area("Predicted List is Here", result, key="predicted_list")
