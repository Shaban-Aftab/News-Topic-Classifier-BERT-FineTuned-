import torch
import torch.nn as nn
from transformers import AutoTokenizer
from preprocess import get_dataloaders
from model import NewsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

MODEL_DIR = "saved_models/bert-agnews"
TOKENIZER_DIR = "saved_models/tokenizer"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def evaluate_model():
    """
    Performs scientific evaluation on the test dataset.
    Loads custom model weights, runs batched inference, computes metrics,
    and analyzes high-confidence classification errors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device.type.upper()}")
    
    # 1. Load tokenizer and high-performance test dataloader
    print("Loading tokenizer and test dataset...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    _, test_loader = get_dataloaders(batch_size=32)
    
    # 2. Reconstruct custom model structure and load state dictionary weights
    print("\nReconstructing model architecture and loading weights...")
    model = NewsClassifier(num_classes=4, freeze_backbone=False)
    weights_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No trained model found at {weights_path}. Please run train.py first!")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_raw_texts = []
    
    # Extract raw texts from original dataset to perform diagnostics later
    from datasets import load_dataset
    raw_test_dataset = load_dataset("ag_news")["test"]
    raw_texts = raw_test_dataset["text"]
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 3. Print formal Scientific Report
    print(f"\n{'='*55}")
    print("                EVALUATION METRICS REPORT")
    print(f"{'='*55}")
    print(classification_report(all_labels, all_preds, target_names=LABELS, digits=4))
    print(f"{'='*55}")
    
    print("\nConfusion Matrix:")
    conf_mat = confusion_matrix(all_labels, all_preds)
    print(f"            Predicted: {', '.join(LABELS)}")
    for i, row in enumerate(conf_mat):
        print(f"Actual {LABELS[i]:<10}: {row}")
        
    # --- Diagnostics & Error Identification ---
    # Find the top 5 highly confident but incorrect predictions
    print(f"\n{'='*55}")
    print("        TOP 5 HIGH-CONFIDENCE CLASSIFICATION ERRORS")
    print(f"{'='*55}")
    
    incorrect_indices = np.where(all_preds != all_labels)[0]
    
    # Extract probabilities corresponding to the incorrect predictions made
    incorrect_predicted_probs = [all_probs[idx][all_preds[idx]] for idx in incorrect_indices]
    
    # Sort in descending order of predicted probability (highest confidence errors first)
    sorted_error_positions = np.argsort(incorrect_predicted_probs)[::-1]
    
    for rank in range(min(5, len(sorted_error_positions))):
        err_pos = incorrect_indices[sorted_error_positions[rank]]
        true_lbl = LABELS[all_labels[err_pos]]
        pred_lbl = LABELS[all_preds[err_pos]]
        confidence = all_probs[err_pos][all_preds[err_pos]]
        headline = raw_texts[err_pos]
        
        print(f"\n{rank + 1}. Confidence: {confidence:.2%}")
        print(f"   True Category     : {true_lbl}")
        print(f"   Predicted Category: {pred_lbl}")
        print(f"   News Headline     : \"{headline}\"")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    evaluate_model()
