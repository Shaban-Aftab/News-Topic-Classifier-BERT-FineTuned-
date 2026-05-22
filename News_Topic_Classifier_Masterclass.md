# News Topic Classifier from First Principles in PyTorch: Complete Masterclass

Welcome to the ultimate learning guide for building, training, and deploying a state-of-the-art **News Topic Classifier** using **PyTorch** and **BERT** from absolute first principles. 

In this masterclass, you will move beyond high-level wrappers like Hugging Face's `Trainer` and write your own custom PyTorch datasets, dataloaders, model architectures, training loops, metrics, and serving pipelines.

---

## Table of Contents
1. [Lesson 1: Data Foundations & High-Performance Data Pipelines](#lesson-1-data-foundations--high-performance-data-pipelines)
2. [Lesson 2: Demystifying Transformer Architecture & Custom Heads](#lesson-2-demystifying-transformer-architecture--custom-heads)
3. [Lesson 3: Custom PyTorch Training Loop & Optimization Mechanics](#lesson-3-custom-pytorch-training-loop--optimization-mechanics)
4. [Lesson 4: Scientific Evaluation & Diagnostics](#lesson-4-scientific-evaluation--diagnostics)
5. [Lesson 5: Production Deployment & Model Optimization](#lesson-5-production-deployment--model-optimization)

---

# Lesson 1: Data Foundations & High-Performance Data Pipelines

### Learning Objectives
* Understand subword tokenization (WordPiece) and vocabulary mapping.
* Explain the role of attention masks and special tokens (`[CLS]`, `[SEP]`, `[PAD]`).
* Implement a custom PyTorch `Dataset` and `DataLoader` from scratch.
* Build a custom collate function to execute **Dynamic Padding** (algorithmic improvement) and configure **Memory Pinning** (engineering improvement).

### Theory: Text Processing in Modern NLP
* **WordPiece Tokenization:** Modern NLP models split words into subwords (e.g., `"unaffable"` → `["un", "##aff", "##able"]`). This allows the model to handle unseen words gracefully.
* **Special Tokens:**
  * `[CLS]` (ID 101): Placed at the sequence start. Its final state is used for classification.
  * `[SEP]` (ID 102): Placed at the end of each sequence.
  * `[PAD]` (ID 0): Appended to shorter sequences to form even batches.
* **Attention Mask:** A binary tensor where `1` indicates real text and `0` indicates padding, telling the self-attention mechanism to completely ignore padded elements.

### Code Snippet: PyTorch Dataset & Dynamic Collation DataLoader
```python
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name="bert-base-uncased", max_length=128):
        # HTML Cleaning Regex (Algorithmic Improvement)
        self.texts = [re.sub(r'<[^>]+>', ' ', str(t)).strip() for t in texts]
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize sequence without static padding
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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])

        # Dynamically pad sequences in the current batch to the maximum sequence length in that batch
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

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
```

#### Explanation
1. `class NewsDataset(Dataset)`: Inherits from PyTorch's base `Dataset` class, forcing us to override `__len__` and `__getitem__`.
2. `self.texts = [re.sub(r'<[^>]+>', ' ', str(t)).strip() for t in texts]`: Regular expression preprocessor that strips out HTML tags from headlines.
3. `padding=False`: Under `__getitem__`, we retrieve individual samples. We disable padding here so that we only encode raw length tokens, postponing padding until batching.
4. `class DynamicPaddingCollator`: A custom collate function passed to the `DataLoader` that intercepts batch lists and pads them dynamically to the maximum sequence length *in that batch*, saving massive computation.

### Verification Test
```python
if __name__ == "__main__":
    dummy_texts = ["Sports headline!", "Short text.", "This is a significantly longer headline for Business news."]
    dummy_labels = [1, 0, 2]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = NewsDataset(dummy_texts, dummy_labels)
    collator = DynamicPaddingCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator, pin_memory=True)
    
    for batch in loader:
        print("Batch input_ids shape:", batch["input_ids"].shape)
        assert batch["input_ids"].shape == batch["attention_mask"].shape
    print("✓ Data pipeline and dynamic padding collator verified!")
```

---

# Lesson 2: Demystifying Transformer Architecture & Custom Heads

### Learning Objectives
* Understand what hidden states represent inside deep Transformer networks.
* Extract the special `[CLS]` token contextual vector to represent whole sentences.
* Assemble a custom multi-layer classification head in native PyTorch (`nn.Module`).
* Implement backbone layer parameter freezing and multi-layer feature concatenation.

### Theory: BERT Hidden States & Pooling
* **The CLS Representation:** The `[CLS]` token (index `0` in input sequences) passes through all 12 self-attention layers of BERT. At the output, it acts as a semantic summary of the entire sentence.
* **Concatenating Last 4 Layers:** Instead of relying on just the final layer, concatenating the last 4 layers (`768 * 4 = 3072` dimensions) captures both syntactic detail and high-level semantics.

### Code Snippet: Custom model inheriting from `nn.Module`
```python
import torch
import torch.nn as nn
from transformers import AutoModel

class NewsClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=4, freeze_backbone=False):
        super(NewsClassifier, self).__init__()
        # Load raw BERT encoder backbone
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.bert.config.hidden_size
        
        # Concatenated classification head (3072 inputs -> 4 outputs)
        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_classes)
        )
        
        # Backbone parameter freezing
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True # Crucial parameter
        )
        
        # Extract CLS token from last 4 layers
        hidden_states = outputs.hidden_states
        cls_9 = hidden_states[-4][:, 0, :]   # Shape: (Batch, 768)
        cls_10 = hidden_states[-3][:, 0, :]  # Shape: (Batch, 768)
        cls_11 = hidden_states[-2][:, 0, :]  # Shape: (Batch, 768)
        cls_12 = hidden_states[-1][:, 0, :]  # Shape: (Batch, 768)
        
        # Concatenate features
        pooled_output = torch.cat((cls_12, cls_11, cls_10, cls_9), dim=-1) # Shape: (Batch, 3072)
        
        logits = self.classification_head(pooled_output)
        return logits
```

#### Explanation
1. `AutoModel.from_pretrained(...)`: Loads raw BERT without built-in classification layers.
2. `output_hidden_states=True`: Instructs BERT to return all intermediate representations.
3. `hidden_states[-1][:, 0, :]`: Slices the batch output at token index `0` (the `[CLS]` token) across layers.
4. `torch.cat(..., dim=-1)`: Combines the last 4 layers of BERT to capture multi-level semantics.

---

# Lesson 3: Custom PyTorch Training Loop & Optimization Mechanics

### Learning Objectives
* Master writing native PyTorch training loops (`zero_grad()`, `backward()`, `step()`).
* Understand loss computation with **Label Smoothing** regularization.
* Optimize speed using **Automatic Mixed Precision (AMP)** and **Gradient Accumulation**.
* Build a linear warmup + cosine decay scheduler.

### Theory: Optimization Techniques
* **Label Smoothing:** Replaces hard target vectors `[0.0, 1.0, 0.0, 0.0]` with smoothed distributions `[0.025, 0.925, 0.025, 0.025]` (at $\alpha=0.1$). This prevents overconfidence and increases test F1-score generalization.
* **Automatic Mixed Precision (AMP):** Performs forward passes in 16-bit float (`fp16`) instead of 32-bit, roughly doubling GPU throughput.

### Code Snippet: PyTorch Custom Optimization Pipeline
```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) # Label Smoothing
    
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 1. AMP Autocast Context
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss = loss / 2 # Simulating gradient accumulation (steps=2)
            
        # 2. Backpropagation with scaling
        if device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # 3. Parameter Update
        if (step + 1) % 2 == 0 or (step + 1) == len(loader):
            if device.type == "cuda":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad()
```

#### Explanation
1. `nn.CrossEntropyLoss(label_smoothing=0.1)`: Applies regularization to target label vectors.
2. `with torch.cuda.amp.autocast()`: Executes the model forward pass in half-precision (16-bit float) on GPU.
3. `scaler.scale(loss).backward()`: Scales gradients to prevent arithmetic underflow.
4. `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`: Limits gradient norm to `1.0` to prevent exploding gradients.

---

# Lesson 4: Scientific Evaluation & Diagnostics

### Learning Objectives
* Build evaluation functions to compute Accuracy, Precision, Recall, and Weighted F1-Score.
* Perform error diagnostics (analyzing the most confident misclassifications).
* Automate validation-based checkpointing.

### Code Snippet: Confidence Diagnostics Pipeline
```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def run_diagnostics(all_preds, all_labels, all_probs, raw_texts, labels_list):
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    
    # Generate formal report
    print(classification_report(labels, preds, target_names=labels_list, digits=4))
    
    # Retrieve index of wrong predictions
    incorrect_indices = np.where(preds != labels)[0]
    
    # Sort wrong predictions based on confidence score assigned to incorrect classes
    incorrect_predicted_probs = [probs[idx][preds[idx]] for idx in incorrect_indices]
    sorted_error_positions = np.argsort(incorrect_predicted_probs)[::-1]
    
    print("\n--- Top Confident Errors ---")
    for i in range(min(3, len(sorted_error_positions))):
        err_idx = incorrect_indices[sorted_error_positions[i]]
        print(f"Confidence: {probs[err_idx][preds[err_idx]]:.2%} | True: {labels_list[labels[err_idx]]} | Pred: {labels_list[preds[err_idx]]}")
        print(f"  Headline: \"{raw_texts[err_idx]}\"\n")
```

#### Explanation
1. `np.where(preds != labels)[0]`: Extracts indices of misclassified samples.
2. `probs[idx][preds[idx]]`: Accesses the confidence score corresponding to the incorrect prediction.
3. `np.argsort(...)[::-1]`: Sorts confident errors in descending order to identify model edge cases.

---

# Lesson 5: Production Deployment & Model Optimization

### Learning Objectives
* Quantize model weights dynamically to reduce size by 4x.
* Deploy your custom model using an interactive Streamlit interface.
* Benchmark inference latency profiling in real-time.

### Theory: CPU Dynamic INT8 Quantization
Dynamic quantization converts linear layer parameters from 32-bit floating point numbers (`float32`) to 8-bit integers (`qint8`).
* **Footprint reduction:** Shrinks the model from 440MB to ~110MB.
* **Speed:** Accelerates CPU inference by 2x-3x.

### Code Snippet: PyTorch Dynamic Quantization & Streamlit App
```python
import streamlit as st
import torch
import torch.nn as nn
from model import NewsClassifier

@st.cache_resource
def load_optimized_pipeline(weights_path):
    model = NewsClassifier(num_classes=4)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    # Dynamic INT8 Quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model
```

#### Explanation
1. `@st.cache_resource`: Caches the model configuration in Streamlit memory so it is loaded and quantized only once.
2. `torch.quantization.quantize_dynamic(...)`: PyTorch's native dynamic quantizer that optimizes linear layers for CPU execution.
