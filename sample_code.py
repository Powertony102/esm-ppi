import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

class CrossAttentionBlock(nn.Module):
    """
    交叉注意力模块：让两个蛋白质序列的特征进行交互。
    Protein A (Query) 去查询 Protein B (Key, Value) 的信息。
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        """
        Args:
            query: [batch_size, seq_len_A, hidden_dim]
            key_value:
            key_padding_mask: (True indicates padding)
        """
        # Multihead Attention
        attn_output, _ = self.multihead_attn(
            query=query, 
            key=key_value, 
            value=key_value, 
            key_padding_mask=key_padding_mask
        )
        
        # Residual Connection & Norm
        output = self.norm(query + self.dropout(attn_output))
        return output

class SE_CAI_Model(nn.Module):
    """
    Siamese ESM-2 Cross-Attention Interaction Model
    """
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", use_lora=True):
        super().__init__()
        
        # 1. Load Pre-trained ESM-2
        print(f"Loading ESM-2 backbone: {model_name}...")
        self.esm_backbone = AutoModel.from_pretrained(model_name)
        
        # 2. Apply LoRA (Parameter Efficient Fine-Tuning)
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=8,            # LoRA秩
                lora_alpha=16, 
                lora_dropout=0.1,
                target_modules=["query", "key", "value"] # 仅在Attention层应用
            )
            self.esm_backbone = get_peft_model(self.esm_backbone, peft_config)
            self.esm_backbone.print_trainable_parameters()
            
        self.hidden_dim = self.esm_backbone.config.hidden_size # 1280 for 650M
        
        # 3. Cross Attention Layers
        # A looks at B
        self.cross_attn_A2B = CrossAttentionBlock(self.hidden_dim)
        # B looks at A
        self.cross_attn_B2A = CrossAttentionBlock(self.hidden_dim)
        
        # 4. Classification Head (MLP)
        # Input dim calculation:
        # We pool (Max+Mean) for both A and B -> 2 * hidden_dim each
        # We fuse [u, v, |u-v|, u*v] -> 4 * (feature_dim)
        # Total = 4 * (2 * 1280) = 10240
        feature_dim = self.hidden_dim * 2 
        fusion_input_dim = feature_dim * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1) # Binary classification output (logits)
        )

    def forward_protein_features(self, input_ids, attention_mask):
        """Extract features using ESM-2 backbone"""
        outputs = self.esm_backbone(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: [batch, seq_len, 1280]
        return outputs.last_hidden_state

    def pool_features(self, sequence_output, attention_mask):
        """
        Global Pooling: Concatenate Max Pooling and Mean Pooling
        Masking is important to ignore padding tokens.
        """
        # Create mask for pooling (expand to hidden dim)
        # attention_mask: [batch, seq_len] -> [batch, seq_len, 1]
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        
        # Mean Pooling
        sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooling = sum_embeddings / sum_mask
        
        # Max Pooling (Set padding to large negative number)
        sequence_output_masked = sequence_output.clone()
        sequence_output_masked[mask_expanded == 0] = -1e9
        max_pooling, _ = torch.max(sequence_output_masked, 1)
        
        return torch.cat([mean_pooling, max_pooling], dim=1) # [batch, 2560]

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        # 1. Independent Encoding (Siamese)
        # [batch, len_A, 1280]
        feat_A = self.forward_protein_features(input_ids_A, attention_mask_A)
        #
        feat_B = self.forward_protein_features(input_ids_B, attention_mask_B)
        
        # 2. Cross-Attention Interaction
        # Note: key_padding_mask in PyTorch MultiheadAttention expects True for PADDING
        # Transformers attention_mask uses 0 for PADDING, 1 for REAL.
        # So we invert: mask_A == 0 -> True
        padding_mask_A = (attention_mask_A == 0)
        padding_mask_B = (attention_mask_B == 0)
        
        # A queries B (inject info from B into A)
        feat_A_enhanced = self.cross_attn_A2B(query=feat_A, key_value=feat_B, key_padding_mask=padding_mask_B)
        
        # B queries A (inject info from A into B)
        feat_B_enhanced = self.cross_attn_B2A(query=feat_B, key_value=feat_A, key_padding_mask=padding_mask_A)
        
        # 3. Pooling (Sequence -> Vector)
        vector_A = self.pool_features(feat_A_enhanced, attention_mask_A) # [batch, 2560]
        vector_B = self.pool_features(feat_B_enhanced, attention_mask_B) # [batch, 2560]
        
        # 4. Heuristic Fusion
        # Concatenate: [u, v, |u-v|, u*v]
        diff_sim = torch.abs(vector_A - vector_B)
        prod_sim = vector_A * vector_B
        
        fused_vector = torch.cat(, dim=1) # [batch, 10240]
        
        # 5. Classification
        logits = self.classifier(fused_vector)
        return logits

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, targets: binary labels (0 or 1)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Prevents nans
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==========================================
# Example Usage Snippet
# ==========================================
if __name__ == "__main__":
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/esm2_t33_650M_UR50D"
    
    # 1. Initialize Model
    model = SE_CAI_Model(model_name=model_name, use_lora=True).to(device)
    
    # 2. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. Mock Data Batch
    seqs_A =
    seqs_B =
    labels = torch.tensor([[1.0], [0.0]]).to(device) # Labels must be float for BCE
    
    # 4. Tokenization
    # Note: Using dynamic padding via the tokenizer
    inputs_A = tokenizer(seqs_A, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
    inputs_B = tokenizer(seqs_B, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
    
    # 5. Forward Pass
    logits = model(
        input_ids_A=inputs_A['input_ids'],
        attention_mask_A=inputs_A['attention_mask'],
        input_ids_B=inputs_B['input_ids'],
        attention_mask_B=inputs_B['attention_mask']
    )
    
    print(f"Logits shape: {logits.shape}") # Should be [1, 2]
    
    # 6. Loss Calculation
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, labels)
    
    print(f"Loss: {loss.item()}")
    
    # 7. Prediction
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()
    print(f"Probabilities: {probs.detach().cpu().numpy()}")
    print(f"Predictions: {preds.detach().cpu().numpy()}")