# News-Topic-Classifier-Using-BERT

Fine-tune a BERT model to classify news headlines into topic categories (World, Sports, Business, Sci/Tech) using the AG News dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Running on Google Colab (Recommended)

If you don't have a powerful machine with a GPU, you can run this project for free on **Google Colab with a T4 GPU**. The training script is optimized for it.

1. Upload the project files to Google Colab or connect your GitHub repo
2. Select runtime: **Runtime → Change runtime type → T4 GPU**
3. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
4. Run training:
   ```bash
   !python train.py
   ```

Training on a T4 GPU takes approximately 10-15 minutes.

## Usage

### Train the model

```bash
python train.py
```

This will:
1. Load and tokenize the AG News dataset
2. Fine-tune `bert-base-uncased` for 3 epochs with mixed precision (FP16)
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
