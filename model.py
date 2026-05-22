import torch
import torch.nn as nn
from transformers import AutoModel

class NewsClassifier(nn.Module):
    """
    Custom Sequence Classification Model built from first principles in PyTorch.
    Loads a base pretrained BERT encoder, extracts context representations,
    and applies a custom feed-forward neural network for topic classification.
    """
    def __init__(self, model_name="bert-base-uncased", num_classes=4, freeze_backbone=False):
        super(NewsClassifier, self).__init__()
        
        # 1. Load raw BERT encoder backbone (outputs hidden states, no head)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 2. Extract hidden dimension size dynamically (usually 768 for BERT base)
        self.hidden_dim = self.bert.config.hidden_size
        
        # --- Algorithmic Improvement: Last 4 Layers Concatenation Head ---
        # Instead of using just the final CLS vector (768), we concatenate the CLS tokens
        # from the last 4 layers (768 * 4 = 3072) to capture rich, multi-level semantic representations.
        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_classes)
        )
        
        # --- Engineering Improvement: Selective Backbone Freezing ---
        if freeze_backbone:
            print("Freezing BERT backbone parameters...")
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        # 3. Pass inputs through BERT backbone
        # Output all hidden states by configuring the forward call
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True # Crucial for last 4 layer retrieval
        )
        
        # 4. Extract all layers' hidden states (tuple of 13 tensors: embedding + 12 layers)
        hidden_states = outputs.hidden_states
        
        # 5. Extract and concatenate the [CLS] token (index 0) from the last 4 encoder layers
        cls_layer_12 = hidden_states[-1][:, 0, :] # Shape: (batch_size, 768)
        cls_layer_11 = hidden_states[-2][:, 0, :] # Shape: (batch_size, 768)
        cls_layer_10 = hidden_states[-3][:, 0, :] # Shape: (batch_size, 768)
        cls_layer_9  = hidden_states[-4][:, 0, :] # Shape: (batch_size, 768)
        
        # Concatenate along the feature dimension (dim=-1)
        # Resulting Shape: (batch_size, 3072)
        pooled_output = torch.cat(
            (cls_layer_12, cls_layer_11, cls_layer_10, cls_layer_9), 
            dim=-1
        )
        
        # 6. Run representations through our custom neural classification head
        # Shape: (batch_size, num_classes)
        logits = self.classification_head(pooled_output)
        
        return logits

if __name__ == "__main__":
    print("Testing custom model construction and dimensions...")
    
    # 7. Instantiate model
    model = NewsClassifier(num_classes=4, freeze_backbone=True)
    
    # 8. Create dummy inputs (Batch Size = 2, Sequence Length = 8)
    dummy_input_ids = torch.randint(0, 1000, (2, 8))
    dummy_attention_mask = torch.ones((2, 8), dtype=torch.long)
    
    # 9. Perform forward pass
    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)
        
    print("Dummy input shape:", dummy_input_ids.shape)
    print("Output logits shape:", logits.shape)
    
    # Assert output shape matches expectation
    assert logits.shape == (2, 4), f"Expected shape (2, 4), got {logits.shape}"
    print("\n✓ Model forward pass and dynamic hidden state concatenation verified successfully!")
