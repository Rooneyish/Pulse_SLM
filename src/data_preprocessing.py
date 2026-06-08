import regex as re
import os 
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from torch.nn.utils.rnn import pad_sequence

class WikiDataPipepline:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_path = self.data_dir / 'train.txt'
        self.val_path = self.data_dir / 'val.txt'

        self.start_token, self.stard_id = '<START>', 1
        self.end_token, self.end_id = '<END>', 2

        self.stoi = {}
        self.itos = {}

        self.tokens = []
        self.max_seq_len = 0

    def clean_txt(self, text):
        text = re.sub(r"[^\p{Latin}\p{P}\p{N}\p{Z}]", " ", text) # removes non-Latin characters
        text = text.replace("\\'", "'") # fixes escaped apostrophes
        
        text = re.sub(r"\s+'\s*([sn])", r"'\1", text)   # fixes contractions like " 's" to "'s"
        text = re.sub(r"={1,}\s*(.*?)\s*={1,}", r"\1", text) # removes headers
        text = re.sub(r"\s+([,.:;?!])", r"\1", text)    # fixes punctuation spacing
        text = re.sub(r"\s*@\s*-\s*@\s*", "-", text)    # fixes @-@ hyphens
        text = re.sub(r"\s+", " ", text).strip()        # collapses extra spaces
        text = text.replace("<unk>", "")        # removes <unk> tokens
        text = re.sub(r"\n{2,}", "\n", text)    # collapses multiple newlines into one
        text = re.sub(r"\s+", " ", text)    # collapses multiple spaces into one
        text = text.strip()     # removes leading/trailing whitespace            
        text = text.lower()     # converts to lowercase

        
        return text

    def sentence_tokenize(self, text):
        sentences = sent_tokenize(text)
        return sentences

    def build_vocab(self, sentences):
        self.tokens = []
        corpus = {}

        for sentence in sentences:
            wrapped_sentence = f"{self.start_token} {sentence} {self.end_token}"
            word_list = wrapped_sentence.split()
            
            self.tokens.append(word_list)
            
            for word in word_list:
                if word in corpus:
                    corpus[word] += 1
                else:
                    corpus[word] = 1

        corpus = {k: v for k, v in sorted(corpus.items(), key=lambda item: item[1], reverse=True)}

        corpus.pop(self.start_token, None)
        corpus.pop(self.end_token, None)

        unique_words = list(corpus.keys())

        self.stoi = {"<PAD>": 0, "<UNK>": 1, self.start_token: self.stard_id, self.end_token: self.end_id}
        self.itos = {0: "<PAD>", 1: "<UNK>", self.stard_id: self.start_token, self.end_id: self.end_token}

        for idx, word in enumerate(unique_words, start=3):
            self.stoi[word] = idx
            self.itos[idx] = word
    
    def encode_sentence(self):
        tokenized_sentence = []
        for word_list in self.tokens:
            encoded_seq = [self.stoi.get(word, 1) for word in word_list]
            tokenized_sentence.append(encoded_seq)
        return tokenized_sentence
    
    def construct_tensor(self, tokenized_sentences, is_validation=False):
        sequence = []
        for sentence in tokenized_sentences:
            for i in range(1, len(sentence)):
                sequence.append(sentence[:i+1])

        tensor_seq = [torch.tensor(seq) for seq in sequence]
        padded_sequence = pad_sequence(tensor_seq, batch_first=True, padding_side='left', padding_value=0)

        if not is_validation:
            self.max_seq_len = padded_sequence.shape[1]
        else:
            if padded_sequence.shape[1] < self.max_seq_len:
                pad_amt = self.max_seq_len - padded_sequence.shape[1]
                padded_sequence = torch.nn.functional.pad(padded_sequence, (pad_amt, 0), value=0)
            elif padded_sequence.shape[1] > self.max_seq_len:
                padded_sequence = padded_sequence[:, -self.max_seq_len:]

        X = padded_sequence[:, :-1]
        y = padded_sequence[:, -1]

        return X, y
    
    def run_pipeline(self):
        if not self.train_path.exists() or not self.val_path.exists():
            raise FileNotFoundError("train.txt or val.txt not found in the specified data directory.")
        
        train_raw = self.train_path.read_text(encoding='utf-8')
        val_raw = self.val_path.read_text(encoding='utf-8')
        print("Files loaded successfully. Starting preprocessing...")

        clean_train = self.clean_txt(train_raw)
        clean_val = self.clean_txt(val_raw)
        print("Text cleaning completed. Starting sentence tokenization...")

        train_sentences = self.sentence_tokenize(clean_train)
        val_sentences = self.sentence_tokenize(clean_val)
        print("Sentence tokenization completed. Building vocabulary...")

        self.build_vocab(train_sentences)
        print("Vocabulary built successfully. Encoding sentences...")

        train_tokenized = self.encode_sentence()
        X_train, y_train = self.construct_tensor(train_tokenized)
        print("Training data encoded and tensor constructed. Processing validation data...")

        val_tokens_list = []
        for sentence in val_sentences:
            wrapped_sentence = f"{self.start_token} {sentence} {self.end_token}"
            val_tokens_list.append(wrapped_sentence.split())
        print("Validation sentences tokenized. Encoding validation sentences...")

        self.tokens = val_tokens_list
        val_tokenized = self.encode_sentence()
        X_val, y_val = self.construct_tensor(val_tokenized)
        print("Validation data encoded and tensor constructed. Creating DataLoaders...")

        save_dir = self.data_dir / "processed"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving processed data and pipeline metadata to {save_dir}...")

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
            'max_seq_len': self.max_seq_len
        }, save_dir / "pipeline_meta.pt")

        print("Processed data and metadata saved successfully. Creating DataLoaders...")

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)
        print("DataLoaders created successfully. Pipeline execution completed.")
        return train_loader, val_loader

if __name__ == "__main__":
    data_folder = Path("data")
    
    if os.path.exists(data_folder / "train.txt"):
        pipeline = WikiDataPipepline(data_dir=data_folder, batch_size=32)
        train_dl, val_dl = pipeline.run_pipeline()
        
        for bx, by in train_dl:
            print("\n--- Final Script Verification ---")
            print("Batch X Tensor Shape:", bx.shape)
            print("Batch y Tensor Shape:", by.shape)
            break