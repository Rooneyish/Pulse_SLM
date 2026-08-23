from pathlib import Path
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tqdm import tqdm

data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 4096      
SEQ_LEN = 128          
NUM_SAMPLES = 100_000   

tokenizer_path = data_dir / f"tokenizer_v{VOCAB_SIZE}.json"

print("Loading TinyStories from Hugging Face...")
raw_ds = load_dataset("roneneldan/TinyStories")

train_ds = raw_ds["train"]
val_ds = raw_ds["validation"]

if NUM_SAMPLES:
    print(f"Subsampling to top {NUM_SAMPLES:,} training stories...")
    train_ds = train_ds.select(range(NUM_SAMPLES))  

train_text = train_ds["text"]
val_text = val_ds["text"]

if not tokenizer_path.exists():
    print("Training BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
    )

    tokenizer.train_from_iterator(train_text, trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
else:
    print("Loading existing tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

pad_id = tokenizer.token_to_id("<PAD>")
eos_id = tokenizer.token_to_id("<EOS>")
bos_id = tokenizer.token_to_id("<BOS>")

def process_stories(stories, seq_len):
    all_tokens = []
    print("Encoding text into tokens...")
    encodings = tokenizer.encode_batch(list(stories))  
    for encoded in tqdm(encodings, desc="Processing"):
        ids = encoded.ids
        ids.append(eos_id)
        all_tokens.extend(ids)

    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    num_sequences = (len(tokens_tensor) - 1) // seq_len

    X = tokens_tensor[: num_sequences * seq_len].view(-1, seq_len)
    y = tokens_tensor[1 : num_sequences * seq_len + 1].view(-1, seq_len)

    return X, y

print("\nProcessing Train Set...")
X_train, y_train = process_stories(train_text, SEQ_LEN)

print("\nProcessing Validation Set...")
X_val, y_val = process_stories(val_text, SEQ_LEN)

torch.save(
    {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    },
    data_dir / "processed_data.pt",
)

torch.save(
    {
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "pad_id": pad_id,
        "eos_id": eos_id,
        "bos_id": bos_id,
    },
    data_dir / "pipeline_meta.pt",
)

print("\nDataset preparation complete!")
print(f"Train tensor shape: {X_train.shape} | Val tensor shape: {X_val.shape}")
print(f"Saved artifacts to {data_dir}/")