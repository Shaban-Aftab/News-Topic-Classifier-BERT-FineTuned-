import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from model import NewsClassifier
from huggingface_hub import hf_hub_download
import time
import os

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
DEFAULT_HF_REPO = "shaban/news-topic-classifier-bert" # Placeholder default

@st.cache_resource
def load_optimized_pipeline(model_source: str, hf_repo_id: str = None):
    """
    Loads and optimizes the tokenizer and custom model.
    Supports loading from either local checkpoints or dynamically pulling
    from a specified Hugging Face Hub repository.
    Applies dynamic dynamic INT8 quantization for rapid CPU execution.
    """
    tokenizer_name = "bert-base-uncased"
    weights_path = None
    
    # 1. Determine Model Source and locate state dictionary weights
    if model_source == "Local Checkpoint":
        local_path = "saved_models/bert-agnews/pytorch_model.bin"
        if os.path.exists(local_path):
            weights_path = local_path
            print(f"Loading weights from local path: {weights_path}")
        else:
            st.error("⚠️ Local weights 'saved_models/bert-agnews/pytorch_model.bin' not found! Make sure you train the model first.")
            return None, None
    else:
        # Load dynamically from Hugging Face Hub
        if not hf_repo_id:
            st.error("⚠️ Hugging Face Repository ID is required.")
            return None, None
            
        try:
            st.info(f"Downloading model weights from Hugging Face Hub ({hf_repo_id})...")
            # Fetch pytorch_model.bin weights file from remote repo using hf_hub_download
            weights_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename="pytorch_model.bin"
            )
            print(f"Loaded remote weights cached at: {weights_path}")
            # Try to load tokenizer from same repo, fallback to bert-base-uncased if it fails
            tokenizer_name = hf_repo_id
        except Exception as e:
            st.error(f"❌ Failed to download model from Hugging Face: {e}")
            return None, None

    # 2. Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        # Fallback to base uncased tokenizer if remote custom files are missing
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    # 3. Reconstruct custom sequence classification model structure
    model = NewsClassifier(num_classes=4, freeze_backbone=False)
    
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        st.warning("⚠️ No weights found. Using random initialized parameters for evaluation.")
        
    model.eval()
    
    # 4. Apply Post-Training Dynamic INT8 Quantization (Engineering Optimization)
    # Reduces weight storage by 75% and speeds up CPU prediction time by 2-3x
    print("Quantizing model dynamically for high-performance CPU serving...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    return tokenizer, quantized_model

def predict(text, tokenizer, model):
    """
    Executes a single classification inference pass and measures execution latency.
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    start_time = time.time()
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1)[0]
    latency_ms = (time.time() - start_time) * 1000
    
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    
    return LABELS[pred_idx], confidence, probs.tolist(), latency_ms

def main():
    st.set_page_config(
        page_title="News Topic Classifier", 
        page_icon="📰",
        layout="centered"
    )
    
    st.title("📰 News Topic Classifier")
    st.markdown(
        """
        *Fine-Tuned first-principles BERT model with Dynamic INT8 CPU Optimization.*
        
        Enter a news headline below to classify it instantly into one of four topics: 
        **World, Sports, Business, or Sci/Tech**.
        """
    )
    
    # --- Sidebar Configuration Panel ---
    st.sidebar.title("🛠️ Model Configuration")
    st.sidebar.markdown("Choose where to load the model weights from.")
    
    model_source = st.sidebar.selectbox(
        "Model Weights Source",
        ["Local Checkpoint", "Hugging Face Hub"]
    )
    
    hf_repo_id = None
    if model_source == "Hugging Face Hub":
        hf_repo_id = st.sidebar.text_input(
            "Hugging Face Repo ID",
            placeholder="e.g. username/news-topic-classifier-bert",
            value=""
        ).strip()
        st.sidebar.caption("💡 *Tip: Train the model, upload it using our `upload_to_hf.py` script, then enter your repo name here!*")

    # 2. Load model and tokenizer
    tokenizer, model = None, None
    if model_source == "Hugging Face Hub" and not hf_repo_id:
        st.info("👈 Please enter your Hugging Face Repository ID in the sidebar to load the model.")
    else:
        with st.spinner("Optimizing and Loading Model (Dynamic INT8 Quantization)..."):
            tokenizer, model = load_optimized_pipeline(model_source, hf_repo_id)
            
    # 3. Text area inputs
    user_input = st.text_area("Enter news headline:", height=100, placeholder="e.g., Apple plans to release their new hybrid processors next month...")
    
    if st.button("Classify Headline", type="primary"):
        if not tokenizer or not model:
            st.error("Model not loaded. Configure settings in the sidebar first.")
        elif not user_input.strip():
            st.warning("Please enter a valid text sequence.")
        else:
            with st.spinner("Analyzing text..."):
                label, confidence, all_probs, latency = predict(user_input, tokenizer, model)
            
            # Category Callout UI
            st.success(f"### Predicted Topic: **{label}**")
            
            # Latency and Length Metrics columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence Score", f"{confidence:.2%}")
            with col2:
                st.metric("Inference Speed", f"{latency:.1f} ms")
            with col3:
                st.metric("Character Length", f"{len(user_input)}")
                
            # Probabilities distribution chart
            st.subheader("Probability Distribution")
            for lbl, prob in zip(LABELS, all_probs):
                st.progress(prob, text=f"**{lbl}**: {prob:.2%}")
                
            st.caption("ℹ️ *Speed is optimized using PyTorch Dynamic Quantization, converting FP32 parameters to INT8.*")

if __name__ == "__main__":
    main()
