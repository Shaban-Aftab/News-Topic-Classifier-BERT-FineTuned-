import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

MODEL_DIR = "saved_models/bert-agnews"
TOKENIZER_DIR = "saved_models/tokenizer"
MAX_LENGTH = 128
BATCH_SIZE = 32

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def evaluate_model():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Loading test dataset...")
    dataset = load_dataset("ag_news")
    test_data = dataset["test"]
    
    all_preds = []
    all_labels = []
    
    print("Running inference on test set...")
    for i in range(0, len(test_data), BATCH_SIZE):
        batch_texts = test_data[i:i+BATCH_SIZE]["text"]
        batch_labels = test_data[i:i+BATCH_SIZE]["label"]
        
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(batch_labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    
    print(f"\n{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=LABELS))
    
    return accuracy, f1

if __name__ == "__main__":
    evaluate_model()
