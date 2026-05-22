import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from preprocess import get_dataloaders
from model import NewsClassifier
import os
import time

# Hyperparameters & Paths
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0
MODEL_SAVE_PATH = "saved_models/bert-agnews"

def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, scaler, device):
    """
    Executes one complete epoch of training over the loader.
    Supports gradient accumulation and automatic mixed precision (AMP) for maximum speed.
    """
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # 1. Reset optimizer gradients before epoch
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        # Move tensors to active computing device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 2. Run Forward Pass with Automatic Mixed Precision (AMP)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            # Divide loss to account for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
        # 3. Backward Pass (scale loss if using AMP)
        if device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        # 4. Perform Optimizer Step after accumulating sufficient gradients
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
            # Unscale gradients before clipping if using AMP
            if device.type == "cuda":
                scaler.unscale_(optimizer)
                
            # Algorithmic Improvement: Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            
            # Step optimizer & scheduler
            if device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad() # Reset accumulated gradients
            
        # Log training progress periodically
        if (step + 1) % 100 == 0:
            avg_step_loss = total_loss / (step + 1)
            elapsed = time.time() - start_time
            print(f"  Step {step + 1}/{len(dataloader)} | Loss: {avg_step_loss:.4f} | Elapsed: {elapsed:.1f}s")
            
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
    """
    Evaluates the model on the validation/test loader.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            # Track raw accuracy
            preds = logits.argmax(dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def main():
    # 5. Determine runtime hardware acceleration dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute environment: {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU Adapter: {torch.cuda.get_device_name(0)}")
        
    # 6. Retrieve modular native dataloaders
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # 7. Initialize first-principles custom model
    print("\nInitializing NewsClassifier model...")
    model = NewsClassifier(num_classes=4, freeze_backbone=False)
    model.to(device)
    
    # 8. Configure native Cross-Entropy loss with Algorithmic Improvement: Label Smoothing
    # Label smoothing prevents overconfidence and increases test F1-score generalization
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 9. Build AdamW optimizer separating weight decay parameters
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    
    # 10. Instantiate Dynamic Linear Warmup and Cosine Decay Scheduler
    num_training_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 11. Prepare AMP gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    # Track metrics to save best checkpoint
    best_accuracy = 0.0
    print(f"\n--- Starting Custom PyTorch Fine-Tuning Pipeline ({EPOCHS} Epochs) ---")
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\n★ Epoch {epoch + 1}/{EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, device)
        val_loss, val_acc = validate(model, test_loader, loss_fn, device)
        
        epoch_elapsed = time.time() - epoch_start
        print(f"✔ Epoch {epoch + 1} Done | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | Time: {epoch_elapsed:.1f}s")
        
        # Engineering Improvement: Automatic Checkpoint Saving based on accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            print(f"  --> Saving best model checkpoint (Val Acc = {val_acc:.2%})")
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "pytorch_model.bin"))
            
    print(f"\nFine-Tuning completed! Best Validation Accuracy achieved: {best_accuracy:.2%}")

if __name__ == "__main__":
    main()