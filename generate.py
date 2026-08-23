from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, processors
from src.model import TransformerDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = Path("data/processed")
checkpoint_path = Path("checkpoints/best_slm.pt")

meta = torch.load(data_dir / "pipeline_meta.pt")
tokenizer = Tokenizer.from_file(str(data_dir / f"tokenizer_v{meta['vocab_size']}.json"))
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
vocab_size = meta["vocab_size"]

model = TransformerDecoder(
    vocab_size=vocab_size,         # Vocabulary size (TinyStories)
    d_model=384,          # Hidden representation dimension
    num_layers=8,          # Stacked Transformer decoder blocks
    num_heads=6,           # 8 attention heads (384/ 64=6)
    seq_len=128,       # Sequence length (128)
    head_dim=64,           # Dimension per head
    hidden_n=1536,         # Feedforward network hidden dimension
    dropout_prob=0.2,  
).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(prompt, max_new_tokens=100, temperature=0.7, top_k=40):
    input_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -meta["seq_len"]:]
        
        with torch.no_grad():
            logits = model(x_cond)
            logits = logits[:, -1, :] / temperature  

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_token), dim=1)

        if next_token.item() == meta.get("eos_id"):
            break

    return tokenizer.decode(x[0].tolist())

prompt = "Hi"
print(f"--- Prompt: {prompt} ---\n")
print(generate(prompt, max_new_tokens=120, temperature=0.7, top_k=90))