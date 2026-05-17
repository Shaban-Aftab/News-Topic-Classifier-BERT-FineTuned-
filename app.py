import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "saved_models/bert-agnews"
TOKENIZER_DIR = "saved_models/tokenizer"
MAX_LENGTH = 128

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    pred_idx = probs.argmax(dim=-1).item()
    confidence = probs[0][pred_idx].item()
    
    return LABELS[pred_idx], confidence, probs[0].tolist()

def main():
    st.set_page_config(page_title="News Topic Classifier", page_icon="")
    st.title("News Topic Classifier Using BERT")
    st.write("Enter a news headline to classify it into one of four topics: World, Sports, Business, or Sci/Tech.")
    
    tokenizer, model = load_model_and_tokenizer()
    
    user_input = st.text_area("Enter news headline:", height=100)
    
    if st.button("Classify", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a headline.")
        else:
            with st.spinner("Classifying..."):
                label, confidence, all_probs = predict(user_input, tokenizer, model)
            
            st.success(f"Predicted Topic: **{label}**")
            st.metric("Confidence", f"{confidence:.2%}")
            
            st.subheader("All Probabilities")
            for lbl, prob in zip(LABELS, all_probs):
                st.progress(prob, text=f"{lbl}: {prob:.2%}")

if __name__ == "__main__":
    main()
