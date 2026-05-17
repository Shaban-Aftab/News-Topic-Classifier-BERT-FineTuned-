from datasets import load_dataset
from transformers import AutoTokenizer
import os

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

def load_and_tokenize_data():
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")
    
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    os.makedirs("saved_models", exist_ok=True)
    tokenizer.save_pretrained("saved_models/tokenizer")
    
    print(f"Training samples: {len(tokenized_datasets['train'])}")
    print(f"Test samples: {len(tokenized_datasets['test'])}")
    print("Tokenization complete. Datasets saved in memory.")
    
    return tokenized_datasets

if __name__ == "__main__":
    tokenized_datasets = load_and_tokenize_data()
