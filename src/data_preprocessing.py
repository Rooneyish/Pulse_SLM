from pathlib import Path
import regex as re
import torch
from torch.utils.data import DataLoader, TensorDataset
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize


class WikiDataPipeline:
    def __init__(self, data_dir, batch_size=32, seq_len=64, stride=64):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride = stride

        self.train_path = self.data_dir / 'train.txt'
        self.val_path = self.data_dir / 'val.txt'

        self.start_token, self.start_id = '<START>', 2
        self.end_token, self.end_id = '<END>', 3

        self.stoi = {}
        self.itos = {}

    def clean_txt(self, text):
        text = re.sub(r"[^\p{Latin}\p{P}\p{N}\p{Z}]", " ", text)  # removes non-Latin characters
        text = text.replace("\\'", "'")  # fixes escaped apostrophes

        text = re.sub(r"={1,}\s*(.*?)\s*={1,}", r"\1", text)  # removes headers
        text = text.replace("<unk>", "")  # removes <unk> tokens (before whitespace collapse)

        # Fix contractions BEFORE punctuation-spacing / whitespace collapse
        text = re.sub(r"\s+n't", "n't", text)  # "do n't" -> "don't"
        text = re.sub(r"\s+'(s|re|ve|ll|d|m)\b", r"'\1", text)  # 's, 're, 've, 'll, 'd, 'm

        text = re.sub(r"\s+([,.:;?!])", r"\1", text)  # fixes punctuation spacing
        text = re.sub(r"\s*@\s*-\s*@\s*", "-", text)  # fixes @-@ hyphens

        # Collapse multiple newlines into one BEFORE collapsing all whitespace,
        # otherwise the newline-collapse regex never has anything to match.
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)  # collapse horizontal whitespace only
        text = text.strip().lower()

        return text

    def sentence_tokenize(self, text):
        return sent_tokenize(text)

    def build_vocab(self, sentences, min_count=1, max_vocab_size=None):
        """
        min_count: drop words appearing fewer than this many times (mapped to <UNK> at encode time).
        max_vocab_size: if set, keep only the top-N most frequent words (special tokens don't count
            against this budget).
        """
        corpus = {}
        for sentence in sentences:
            for word in sentence.split():
                corpus[word] = corpus.get(word, 0) + 1

        sorted_words = [
            w for w, c in sorted(corpus.items(), key=lambda item: item[1], reverse=True)
            if c >= min_count
        ]

        if max_vocab_size is not None:
            sorted_words = sorted_words[:max_vocab_size]

        # Define special tokens
        self.stoi = {
            "<PAD>": 0,
            "<UNK>": 1,
            self.start_token: self.start_id,
            self.end_token: self.end_id
        }

        for idx, word in enumerate(sorted_words, start=4):
            self.stoi[word] = idx

        self.itos = {v: k for k, v in self.stoi.items()}

    def encode_sentences_to_stream(self, sentences):
        """Flattens sentences with <START> and <END> tokens into a single contiguous token stream."""
        token_stream = []
        for sentence in sentences:
            tokens = [self.start_token] + sentence.split() + [self.end_token]
            encoded = [self.stoi.get(w, 1) for w in tokens]  
            token_stream.extend(encoded)
        return torch.tensor(token_stream, dtype=torch.long)

    def create_sliding_windows(self, token_stream):
        """
        Chunks token stream into sequences of length (seq_len + 1).
        X = chunk[:-1], Y = chunk[1:]
        """
        total_tokens = len(token_stream)
        chunk_len = self.seq_len + 1  

        if total_tokens < chunk_len:
            raise ValueError(
                f"Token stream too short ({total_tokens} tokens) to form a single "
                f"chunk of length {chunk_len} (seq_len={self.seq_len}). "
                f"Reduce seq_len or provide more data."
            )

        X_list = []
        Y_list = []

        for start in range(0, total_tokens - chunk_len + 1, self.stride):
            chunk = token_stream[start: start + chunk_len]
            X_list.append(chunk[:-1])
            Y_list.append(chunk[1:])

        X_tensor = torch.stack(X_list)
        Y_tensor = torch.stack(Y_list)

        return X_tensor, Y_tensor

    def run_pipeline(self):
        if not self.train_path.exists() or not self.val_path.exists():
            raise FileNotFoundError("train.txt or val.txt not found in the specified data directory.")

        print("Files loaded successfully. Preprocessing text...")
        clean_train = self.clean_txt(self.train_path.read_text(encoding='utf-8'))
        clean_val = self.clean_txt(self.val_path.read_text(encoding='utf-8'))

        print("Tokenizing sentences...")
        train_sentences = self.sentence_tokenize(clean_train)
        val_sentences = self.sentence_tokenize(clean_val)

        print("Building vocabulary from training corpus...")
        self.build_vocab(train_sentences)
        print(f"Vocab Size: {len(self.stoi)}")

        print("Converting sentences into continuous token streams...")
        train_stream = self.encode_sentences_to_stream(train_sentences)
        val_stream = self.encode_sentences_to_stream(val_sentences)

        print(f"Chunking token streams into sequences of length {self.seq_len} (stride={self.stride})...")
        X_train, y_train = self.create_sliding_windows(train_stream)
        X_val, y_val = self.create_sliding_windows(val_stream)

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")

        save_dir = self.data_dir / "processed"
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            },
            save_dir / "processed_data.pt"
        )

        torch.save({
            'stoi': self.stoi,
            'itos': self.itos,
            'seq_len': self.seq_len,
            'vocab_size': len(self.stoi)
        }, save_dir / "pipeline_meta.pt")

        print(f"Processed artifacts saved to {save_dir}.")

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader


if __name__ == "__main__":
    data_folder = Path("data")

    if (data_folder / "train.txt").exists():
        pipeline = WikiDataPipeline(data_dir=data_folder, batch_size=32, seq_len=64, stride=64)
        train_dl, val_dl = pipeline.run_pipeline()

        for bx, by in train_dl:
            print("\n--- Final Script Verification for Training Data ---")
            print("Batch X Tensor Shape:", bx.shape)  
            print("Batch y Tensor Shape:", by.shape)  
            break