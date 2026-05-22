import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import re

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

def clean_text(text: str) -> str:
    """
    Cleans raw input text by stripping HTML tags and normalizing spacing.
    This ensures BERT processes clean headlines instead of raw markup tokens.
    """
    # Remove HTML tags using regex
    clean = re.sub(r'<[^>]+>', ' ', text)
    # Replace multiple spaces with a single space
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

class NewsDataset(Dataset):
    """
    Custom PyTorch Dataset for the AG News classification task.
    Loads raw text and true labels, tokenizing headlines on the fly.
    """
    def __init__(self, texts, labels, tokenizer_name=MODEL_NAME, max_length=MAX_LENGTH):
        self.texts = [clean_text(text) for text in texts]
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize sequence without padding yet (we pad dynamically in the collator)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class DynamicPaddingCollator:
    """
    A custom collate function that intercepts individual dictionary records
    and dynamically pads sequences within each batch to the longest sequence in that batch.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])

        # Pad sequence elements using standard PyTorch padding
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        # Pad attention masks with 0 (ignored by self-attention)
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0
        )

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": labels
        }

def get_dataloaders(batch_size=32, num_workers=0):
    """
    Loads AG News dataset, instantiates custom PyTorch Datasets,
    and returns high-performance DataLoaders with Dynamic Padding.
    """
    print("Loading AG News dataset from Hugging Face...")
    dataset = load_dataset("ag_news")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]
    
    print("Building custom PyTorch Datasets...")
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer_name=MODEL_NAME)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer_name=MODEL_NAME)
    
    collator = DynamicPaddingCollator(tokenizer)
    
    # Save tokenizer configuration locally
    os.makedirs("saved_models/tokenizer", exist_ok=True)
    tokenizer.save_pretrained("saved_models/tokenizer")
    
    print("Creating native DataLoaders with Dynamic Padding and Pin Memory...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2, # Evaluation can use larger batch sizes as it stores no gradients
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset summary: Train size = {len(train_dataset)}, Test size = {len(test_dataset)}")
    return train_loader, test_loader

if __name__ == "__main__":
    # Test DataLoader locally with single batch
    train_loader, _ = get_dataloaders(batch_size=4)
    first_batch = next(iter(train_loader))
    print("\n--- Verification Batch Shape ---")
    print("input_ids shape:", first_batch["input_ids"].shape)
    print("attention_mask shape:", first_batch["attention_mask"].shape)
    print("labels shape:", first_batch["labels"].shape)
    assert first_batch["input_ids"].ndim == 2
    print("\n✓ Preprocessing file verified successfully!")
