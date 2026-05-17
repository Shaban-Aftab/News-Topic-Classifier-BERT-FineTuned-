# News-Topic-Classifier-Using-BERT

Fine-tune a BERT model to classify news headlines into topic categories (World, Sports, Business, Sci/Tech) using the AG News dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python train.py
```

This will:
1. Load and tokenize the AG News dataset
2. Fine-tune `bert-base-uncased` for 3 epochs
3. Save the model and tokenizer to `saved_models/`

### Evaluate the model

```bash
python evaluate.py
```

Outputs accuracy and weighted F1-score on the test set.

### Run the Streamlit app

```bash
streamlit run app.py
```

Opens a web interface where you can enter news headlines and get topic predictions.

## Project Structure

```
preprocess.py    - Data loading and tokenization
train.py         - Model fine-tuning
evaluate.py      - Model evaluation
app.py           - Streamlit deployment
saved_models/    - Trained model and tokenizer
```
