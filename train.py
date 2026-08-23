from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.model import TransformerDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tqdm.write(f"Training SLM on device: {device}")

data_dir = Path("data/processed")
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

data = torch.load(data_dir / "processed_data.pt")
meta = torch.load(data_dir / "pipeline_meta.pt")

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

vocab_size = meta["vocab_size"]
seq_len = meta["seq_len"]

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=32, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val), batch_size=32, shuffle=False
)

print(vocab_size)
model = TransformerDecoder(
    vocab_size=vocab_size,         # Vocabulary size (TinyStories)
    d_model=384,          # Hidden representation dimension
    num_layers=8,          # Stacked Transformer decoder blocks
    num_heads=6,           # 8 attention heads (384/ 64=6)
    seq_len=128,       # Sequence length (128)
    head_dim=64,           # Dimension per head
    hidden_n=1536,         # Feedforward network hidden dimension
    dropout_prob=0.2,      # Dropout probability
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
tqdm.write(f"Model Initialized | Total Parameters: {total_params / 1e6:.2f}M\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)

use_cuda = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

epochs = 25
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD> token (id 0)

# Early Stopping Parameters
best_val_loss = float("inf")
patience = 3
patience_counter = 0

# Training & Validation Loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0

    train_pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1:02d}/{epochs} [Train]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for bx, by in train_pbar:
        bx, by = bx.to(device), by.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_cuda):
            logits = model(bx)
            loss = criterion(logits.view(-1, vocab_size), by.view(-1))

        # Backward Pass with Scaled Gradients
        scaler.scale(loss).backward()

        # Gradient Unscaling & Clipping 
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)

    # Validation Pass
    model.eval()
    total_val_loss = 0.0

    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch+1:02d}/{epochs} [Val]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for bx, by in val_pbar:
            bx, by = bx.to(device), by.to(device)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(bx)
                loss = criterion(logits.view(-1, vocab_size), by.view(-1))
            total_val_loss += loss.item()
            val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_val_loss = total_val_loss / len(val_loader)

    # Checkpoint Saving & Early Stopping Logic
    saved_flag = ""
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0  
        checkpoint_path = checkpoint_dir / "best_slm.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            },
            checkpoint_path,
        )
        saved_flag = " [Checkpoint Saved!]"
    else:
        patience_counter += 1

    tqdm.write(
        f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{saved_flag}"
    )

    if patience_counter >= patience:
        tqdm.write(
            f"\nEarly stopping triggered: Validation loss hasn't improved for {patience} consecutive epochs."
        )
        break

tqdm.write(f"\nTraining complete. Best Validation Loss: {best_val_loss:.4f}")