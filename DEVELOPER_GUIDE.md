# Developer Guide: News Topic Classifier Using BERT

This guide is written for Python developers who are new to NLP, machine learning, and deep learning. It explains every concept, every file, and every line of code in this project.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Key Concepts Explained](#2-key-concepts-explained)
3. [Project Structure](#3-project-structure)
4. [How the Data Flows](#4-how-the-data-flows)
5. [File-by-File Breakdown](#5-file-by-file-breakdown)
6. [Understanding the Libraries](#6-understanding-the-libraries)
7. [How BERT Works (Simplified)](#7-how-bert-works-simplified)
8. [Training Process Explained](#8-training-process-explained)
9. [Evaluation Metrics Explained](#9-evaluation-metrics-explained)
10. [Customization Guide](#10-customization-guide)
11. [Troubleshooting](#11-troubleshooting)
12. [Glossary](#12-glossary)

---

## 1. What This Project Does

This project takes a **news headline** (a short piece of text) and predicts which of **four topics** it belongs to:

| Label | Example Headline |
|-------|-----------------|
| World | "UN holds emergency session on Middle East crisis" |
| Sports | "Lakers defeat Celtics 112-108 in overtime thriller" |
| Business | "Federal Reserve raises interest rates by 0.25%" |
| Sci/Tech | "Apple announces new M3 chip for MacBook Pro" |

It does this using a pre-trained AI model called **BERT**, which has been fine-tuned on the **AG News dataset** (a collection of 120,000+ news articles).

---

## 2. Key Concepts Explained

### What is NLP?
**Natural Language Processing (NLP)** is the field of AI that deals with how computers understand human language. Tasks include classification, translation, summarization, and more.

### What is a Transformer?
A **transformer** is a type of neural network architecture designed to process sequential data like text. Unlike older approaches that read text word-by-word, transformers look at the entire sentence at once and understand relationships between all words simultaneously.

### What is BERT?
**BERT** (Bidirectional Encoder Representations from Transformers) is a specific transformer model created by Google. It was pre-trained on massive amounts of text (Wikipedia, books) and learned a general understanding of language. We "fine-tune" it for our specific task.

### What is Fine-Tuning?
Imagine BERT as a student who has read every book in a library but has never taken a quiz on news topics. **Fine-tuning** is like giving that student a short course on news classification so they can answer our specific questions.

### What is a Tokenizer?
Computers don't understand words. They understand numbers. A **tokenizer** breaks text into pieces (tokens) and converts each piece into a number (token ID).

Example:
```
"Hello world!" → ["hello", "world", "!"] → [7592, 2088, 999]
```

### What is a Tensor?
A **tensor** is just a multi-dimensional array. If you know NumPy arrays, you already know tensors. PyTorch uses tensors instead of NumPy arrays because they can run on GPUs.

### What is a Logit?
A **logit** is the raw, unnormalized output of the model before it's converted to a probability. Think of it as a "score" for each category. We apply **softmax** to convert logits into probabilities that sum to 1.

---

## 3. Project Structure

```
News-Topic-Classifier-Using-BERT/
│
├── preprocess.py          # Loads and tokenizes the dataset
├── train.py               # Fine-tunes BERT on the tokenized data
├── evaluate.py            # Tests the trained model and prints metrics
├── app.py                 # Streamlit web app for live predictions
├── requirements.txt       # Python dependencies
├── README.md              # Quick start guide
├── Task.md                # Project specification
├── .gitignore             # Files to ignore in git
│
└── saved_models/          # Created after training
    ├── tokenizer/         # Saved tokenizer files
    └── bert-agnews/       # Saved trained model files
```

---

## 4. How the Data Flows

```
┌──────────────────────────────────────────────────────────────┐
│                        RAW TEXT INPUT                         │
│              "Fed raises interest rates by 0.25%"              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                       TOKENIZATION                            │
│  ["fed", "raises", "interest", "rates", "by", "0", ".", "25", "%"]
│  → [2345, 8901, 4567, 3456, 2011, 1023, 1012, 5678, 1015]    │
│  (padded/truncated to exactly 128 numbers)                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     BERT MODEL FORWARD PASS                   │
│  Input: 128 token IDs → BERT processes → 4 raw scores (logits)│
│  Example logits: [-2.1, -3.5, 4.8, -1.2]                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                        SOFTMAX                                │
│  Converts logits to probabilities that sum to 1.0             │
│  [0.02, 0.005, 0.97, 0.005]                                  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     PREDICTION OUTPUT                         │
│  Highest probability = 0.97 → Label index 2 → "Business"     │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. File-by-File Breakdown

### 5.1 `preprocess.py` — Data Loading and Tokenization

**Purpose:** Download the AG News dataset, convert text into numbers the model can understand, and save the tokenizer.

#### Constants

```python
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
```

- `MODEL_NAME`: Which version of BERT to use. "uncased" means it converts everything to lowercase.
- `MAX_LENGTH`: Every input to BERT must be the same length. We pad short texts with zeros and cut long texts to 128 tokens.

#### The Main Function: `load_and_tokenize_data()`

**Step 1 — Load the dataset:**
```python
dataset = load_dataset("ag_news")
```
This downloads the AG News dataset from Hugging Face. It contains:
- **train split:** 120,000 articles
- **test split:** 7,600 articles

Each article has two fields:
- `text`: The headline + description
- `label`: An integer 0-3 (World=0, Sports=1, Business=2, Sci/Tech=3)

**Step 2 — Load the tokenizer:**
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```
This downloads the BERT tokenizer, which knows how to split text into the exact tokens BERT expects.

**Step 3 — Define tokenization function:**
```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,       # Cut texts longer than MAX_LENGTH
        padding="max_length",  # Pad short texts to MAX_LENGTH
        max_length=MAX_LENGTH,
    )
```
This function takes a batch of texts and returns a dictionary with:
- `input_ids`: The token IDs (numbers)
- `token_type_ids`: Which tokens belong to which sentence (not important for single-sentence input)
- `attention_mask`: Which tokens are real vs. padding (1 = real, 0 = padding)

**Step 4 — Apply tokenization to the entire dataset:**
```python
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,          # Process in batches (faster)
    remove_columns=["text"], # Remove original text column
)
```
The `.map()` method applies `tokenize_function` to every example. It's like Python's `map()` but optimized for datasets.

**Step 5 — Rename the label column:**
```python
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
```
The Hugging Face `Trainer` expects the column to be named `labels` (plural), not `label`.

**Step 6 — Save the tokenizer:**
```python
os.makedirs("saved_models", exist_ok=True)
tokenizer.save_pretrained("saved_models/tokenizer")
```
Saves the tokenizer configuration so we can reload it later without downloading again.

---

### 5.2 `train.py` — Model Fine-Tuning

**Purpose:** Load BERT, configure training settings, and fine-tune it on the AG News dataset.

#### Constants

```python
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "saved_models/bert-agnews"
NUM_LABELS = 4
EPOCHS = 3
BATCH_SIZE = 16
```

- `NUM_LABELS`: AG News has 4 categories.
- `EPOCHS`: One epoch = one full pass through the training data. 3 epochs means the model sees every training example 3 times.
- `BATCH_SIZE`: How many examples the model processes before updating its weights. Larger = faster but uses more memory.

#### Loading the Model

```python
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)
```
This loads BERT with an added **classification head** on top. The classification head is a small neural network layer that takes BERT's output and produces 4 scores (one per category).

#### Training Arguments

```python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,           # Where to save checkpoints
    evaluation_strategy="epoch",     # Evaluate after each epoch
    save_strategy="epoch",           # Save checkpoint after each epoch
    learning_rate=2e-5,              # How big each weight update is
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    num_train_epochs=3,              # Number of passes through data
    weight_decay=0.01,               # Prevents overfitting
    load_best_model_at_end=True,     # Auto-load the best checkpoint
    metric_for_best_model="accuracy",# Which metric determines "best"
    logging_dir="./logs",            # Where to save training logs
    logging_steps=100,               # Log progress every 100 steps
)
```

**Key concepts:**

- **Learning rate (2e-5 = 0.00002):** Controls how much the model changes its weights after each batch. Too high = model diverges. Too low = training is extremely slow. For fine-tuning BERT, 2e-5 to 5e-5 is the standard range.

- **Weight decay (0.01):** A regularization technique that penalizes large weights. It prevents the model from memorizing the training data (overfitting).

- **Evaluation strategy:** After each epoch, the model is tested on the validation/test set to see how well it generalizes.

#### Metrics Function

```python
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}
```

This function tells the `Trainer` how to evaluate the model:

1. `logits` are the raw scores from the model (e.g., `[-2.1, -3.5, 4.8, -1.2]`)
2. `logits.argmax(axis=-1)` picks the index of the highest score (e.g., index 2)
3. `accuracy_score` compares predictions to true labels
4. `f1_score` with `average="weighted"` computes F1 for each class and weights by class size

#### The Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)
```
The `Trainer` is a Hugging Face utility that handles the entire training loop: batching, forward pass, loss computation, backpropagation, weight updates, evaluation, logging, and checkpointing.

#### Training and Saving

```python
trainer.train()
trainer.save_model(OUTPUT_DIR)
```
`trainer.train()` runs the training loop. `trainer.save_model()` saves the final model weights and configuration.

---

### 5.3 `evaluate.py` — Model Evaluation

**Purpose:** Load the trained model, run it on the test set, and print detailed metrics.

#### Why a Separate Evaluation Script?

The training script already evaluates during training, but this script:
1. Loads the **final saved model** (not from memory)
2. Runs a **full evaluation** on the entire test set
3. Prints a **detailed classification report** (precision, recall, F1 per class)

#### Key Differences from Training

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
This moves the model to the GPU if available. During training, the `Trainer` handles this automatically.

```python
model.eval()
```
This puts the model in **evaluation mode**. Some layers (like dropout) behave differently during training vs. evaluation.

```python
with torch.no_grad():
    outputs = model(**inputs)
```
`torch.no_grad()` disables gradient computation. During evaluation, we don't need gradients, so this saves memory and speeds things up.

#### Batched Inference

```python
for i in range(0, len(test_data), BATCH_SIZE):
    batch_texts = test_data[i:i+BATCH_SIZE]["text"]
    ...
```
The test set has 7,600 examples. Processing them all at once would run out of memory. Instead, we process them in batches of 32.

#### Classification Report

```python
print(classification_report(all_labels, all_preds, target_names=LABELS))
```
This prints a table like:

```
              precision    recall  f1-score   support

       World       0.96      0.97      0.96      1900
      Sports       0.98      0.97      0.98      1900
    Business       0.94      0.93      0.94      1900
    Sci/Tech       0.93      0.94      0.93      1900

    accuracy                           0.95      7600
   macro avg       0.95      0.95      0.95      7600
weighted avg       0.95      0.95      0.95      7600
```

---

### 5.4 `app.py` — Streamlit Web Application

**Purpose:** Provide a user-friendly web interface for live predictions.

#### Caching the Model

```python
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model
```
`@st.cache_resource` tells Streamlit to load the model **only once**. Without this, the model would reload every time the user clicks a button, which would be very slow.

#### Prediction Function

```python
def predict(text, tokenizer, model):
    inputs = tokenizer(text, truncation=True, padding="max_length",
                       max_length=MAX_LENGTH, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    pred_idx = probs.argmax(dim=-1).item()
    confidence = probs[0][pred_idx].item()
    
    return LABELS[pred_idx], confidence, probs[0].tolist()
```

Step by step:
1. Tokenize the input text into numbers
2. Pass through the model to get raw scores (logits)
3. Apply **softmax** to convert logits to probabilities
4. Find the index of the highest probability
5. Map that index to a label name
6. Return the label, confidence, and all probabilities

#### UI Components

```python
st.title("News Topic Classifier Using BERT")
user_input = st.text_area("Enter news headline:", height=100)
if st.button("Classify", type="primary"):
    ...
    st.success(f"Predicted Topic: **{label}**")
    st.metric("Confidence", f"{confidence:.2%}")
    st.progress(prob, text=f"{lbl}: {prob:.2%}")
```
These are Streamlit UI components that create the web interface.

---

## 6. Understanding the Libraries

### 6.1 PyTorch (`torch`)

PyTorch is the deep learning framework that powers everything. Key concepts:

| Concept | What It Is | Python Analogy |
|---------|-----------|----------------|
| Tensor | Multi-dimensional array | NumPy array |
| `torch.no_grad()` | Disables gradient tracking | Read-only mode |
| `.to(device)` | Move data to CPU/GPU | Moving data to different storage |
| `.item()` | Extract a Python number from a tensor | `int(array[0])` |
| `argmax(dim=-1)` | Find index of max value | `list.index(max(list))` |

### 6.2 Hugging Face Transformers

| Class | Purpose |
|-------|---------|
| `AutoTokenizer` | Automatically picks the right tokenizer for a model |
| `AutoModelForSequenceClassification` | Loads a model with a classification head |
| `TrainingArguments` | Configuration object for training hyperparameters |
| `Trainer` | Handles the entire training loop |

### 6.3 Hugging Face Datasets

| Method | Purpose |
|--------|---------|
| `load_dataset("ag_news")` | Downloads and prepares the dataset |
| `.map(function)` | Applies a function to every example |
| `.rename_column(old, new)` | Renames a column |

### 6.4 Scikit-Learn

| Function | Purpose |
|----------|---------|
| `accuracy_score(y_true, y_pred)` | Fraction of correct predictions |
| `f1_score(y_true, y_pred, average="weighted")` | Harmonic mean of precision and recall |
| `classification_report(...)` | Detailed per-class metrics table |

### 6.5 Streamlit

| Function | Purpose |
|----------|---------|
| `st.title()` | Page title |
| `st.text_area()` | Multi-line text input |
| `st.button()` | Clickable button |
| `st.success()` | Green success message |
| `st.metric()` | Display a number with label |
| `st.progress()` | Progress bar (used here as a probability bar) |
| `@st.cache_resource` | Cache expensive operations |

---

## 7. How BERT Works (Simplified)

### The Big Picture

BERT reads text and produces a **representation** (a vector of numbers) that captures the meaning of the text. Here's a simplified view:

```
Input: "The bank of the river was muddy"
                │
                ▼
        ┌───────────────┐
        │   Tokenizer   │  "the" "bank" "of" "the" "river" "was" "muddy"
        │               │  → [101, 1996, 2924, 1997, 1996, 3303, 2001, 9038, 102]
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  Embedding    │  Convert each token ID to a 768-dimensional vector
        │    Layer      │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  Transformer  │  12 layers of self-attention + feed-forward networks
        │   Encoder     │  Each layer refines the representation
        │   (×12)       │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  [CLS] Token  │  Special token whose output represents the whole sentence
        │   Output      │  → 768-dimensional vector
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Classification│  Linear layer: 768 → 4
        │     Head      │  → [score_world, score_sports, score_business, score_scitech]
        └───────────────┘
```

### Self-Attention (The Key Innovation)

Self-attention lets each word "look at" every other word in the sentence and decide how much attention to pay to it.

Example: In "The **bank** of the **river** was muddy", the word "bank" pays attention to "river" and understands it means the side of a river, not a financial institution.

### Why "Bidirectional"?

Older models read text left-to-right or right-to-left. BERT reads the entire sentence at once, so it understands context from both directions simultaneously.

---

## 8. Training Process Explained

### What Happens During One Training Step

```
1. Take a batch of 16 examples
2. Tokenize them → input_ids, attention_mask
3. Forward pass through BERT → logits (raw scores)
4. Compute loss (cross-entropy between logits and true labels)
5. Backward pass → compute gradients
6. Optimizer updates weights using gradients
7. Repeat with next batch
```

### Loss Function: Cross-Entropy

Cross-entropy measures how far the model's predictions are from the correct answer.

Example:
- True label: Business (index 2)
- Model's softmax output: `[0.05, 0.02, 0.88, 0.05]`
- Loss: `-log(0.88) = 0.128` (low loss = good prediction)

If the model was wrong:
- True label: Business (index 2)
- Model's softmax output: `[0.60, 0.05, 0.20, 0.15]`
- Loss: `-log(0.20) = 1.609` (high loss = bad prediction)

The optimizer's job is to minimize this loss across all training examples.

### What "Epochs" Mean

- **1 epoch** = the model has seen every training example once
- With 120,000 examples and batch size 16: 120,000 / 16 = 7,500 steps per epoch
- With 3 epochs: 7,500 × 3 = 22,500 total training steps

### Why a Small Learning Rate?

BERT is already smart from pre-training. We only need to **adjust** it slightly for our task. A large learning rate would "unlearn" what BERT already knows. That's why we use 2e-5 (0.00002) instead of typical values like 0.001.

---

## 9. Evaluation Metrics Explained

### Accuracy

```
Accuracy = (Number of correct predictions) / (Total predictions)
```

Simple but can be misleading if classes are imbalanced.

### Precision

```
Precision = (True Positives) / (True Positives + False Positives)
```

Of all the times the model predicted "Sports", how many were actually Sports?

### Recall

```
Recall = (True Positives) / (True Positives + False Negatives)
```

Of all the actual Sports articles, how many did the model correctly identify?

### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The harmonic mean of precision and recall. A single number that balances both.

### Weighted Average

```
Weighted F1 = (F1_class0 × n_class0 + F1_class1 × n_class1 + ...) / total
```

Each class's F1 is weighted by how many examples it has. This matters when classes are imbalanced.

---

## 10. Customization Guide

### Change the Number of Training Epochs

In `train.py`, line 9:
```python
EPOCHS = 3  # Change to 4 or 5 for potentially better results
```

### Change the Batch Size

In `train.py`, line 10:
```python
BATCH_SIZE = 16  # Reduce to 8 if you get out-of-memory errors
```

### Change the Learning Rate

In `train.py`, line 25:
```python
learning_rate=2e-5,  # Try 3e-5 or 5e-5
```

### Use a Different Model

In `preprocess.py` and `train.py`, change:
```python
MODEL_NAME = "bert-base-uncased"
# Other options:
# MODEL_NAME = "bert-large-uncased"   # Bigger, slower, more accurate
# MODEL_NAME = "distilbert-base-uncased"  # Smaller, faster, slightly less accurate
# MODEL_NAME = "roberta-base"         # Different architecture, often better
```

### Add More Categories

If you have a dataset with different labels, change:
```python
NUM_LABELS = 4  # Change to your number of classes
```

And update `LABELS` in `evaluate.py` and `app.py`.

### Change the Maximum Sequence Length

In `preprocess.py`, line 6:
```python
MAX_LENGTH = 128  # Increase for longer texts, decrease for speed
```

Note: BERT's maximum is 512 tokens.

### Use GPU Training

The `Trainer` automatically uses GPU if available. To force CPU:
```python
training_args = TrainingArguments(
    ...
    no_cuda=True,  # Force CPU
)
```

---

## 11. Troubleshooting

### Out of Memory (OOM) Error

**Cause:** Batch size too large for your GPU/CPU memory.

**Fix:** Reduce `BATCH_SIZE` in `train.py`:
```python
BATCH_SIZE = 8  # or even 4
```

### Slow Training

**Cause:** Running on CPU instead of GPU, or batch size too small.

**Fix:** 
- Use a GPU (Google Colab free tier works)
- Increase `BATCH_SIZE` if memory allows
- Use `distilbert-base-uncased` instead of `bert-base-uncased`

### "CUDA out of memory" Error

**Fix:**
```python
BATCH_SIZE = 4
```
Or enable gradient accumulation in `TrainingArguments`:
```python
gradient_accumulation_steps=4,  # Simulates larger batch size
```

### Model Not Learning (Loss Not Decreasing)

**Possible causes:**
- Learning rate too high or too low
- Labels are incorrect
- Dataset is too small

**Fix:** Try `learning_rate=3e-5` or `learning_rate=1e-5`.

### Import Errors

**Fix:** Make sure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Streamlit App Not Finding Model

**Cause:** You haven't trained the model yet, or the model is saved in a different location.

**Fix:** Run `python train.py` first, then check that `saved_models/bert-agnews/` exists.

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **AG News** | A dataset of 127,600 news articles in 4 categories |
| **Attention** | Mechanism that lets the model focus on relevant parts of input |
| **Backpropagation** | Algorithm that computes gradients for weight updates |
| **Batch** | A group of examples processed together |
| **BERT** | Bidirectional Encoder Representations from Transformers |
| **Classification Head** | Final layer that maps model output to class scores |
| **Cross-Entropy Loss** | Loss function for classification tasks |
| **Dropout** | Randomly disables neurons during training to prevent overfitting |
| **Embedding** | Dense vector representation of a token |
| **Epoch** | One full pass through the training data |
| **Fine-Tuning** | Training a pre-trained model on a specific task |
| **Forward Pass** | Computing the model's output from input |
| **Gradient** | Direction and magnitude of weight change needed to reduce loss |
| **Hugging Face** | Company/platform providing transformers, datasets, and model hub |
| **Logit** | Raw, unnormalized model output score |
| **Logits** | Plural of logit |
| **Loss** | How wrong the model's prediction is |
| **Optimizer** | Algorithm that updates weights (e.g., AdamW) |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Padding** | Adding zeros to make all inputs the same length |
| **Pre-training** | Initial training on massive general text data |
| **Precision** | Of predicted positives, how many are truly positive |
| **Recall** | Of actual positives, how many were correctly predicted |
| **Regularization** | Techniques to prevent overfitting |
| **Self-Attention** | Attention mechanism within transformers |
| **Softmax** | Function that converts logits to probabilities |
| **Step** | One batch processed and one weight update |
| **Tensor** | Multi-dimensional array (PyTorch's data structure) |
| **Token** | A piece of text (word, subword, or character) |
| **Tokenizer** | Converts text to tokens and then to numbers |
| **Transformer** | Neural network architecture based on self-attention |
| **Truncation** | Cutting off text that exceeds maximum length |
| **Weight Decay** | Regularization technique that penalizes large weights |
