import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from preprocess import load_and_tokenize_data
import os

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "saved_models/bert-agnews"
NUM_LABELS = 4
EPOCHS = 3
BATCH_SIZE = 16

def train_model():
    tokenized_datasets = load_and_tokenize_data()
    
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=100,
    )
    
    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": acc, "f1": f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    print("Training complete!")
    return trainer

if __name__ == "__main__":
    trainer = train_model()
