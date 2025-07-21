import re
from collections import Counter

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    vocab = {word: i + 1 for i, (word, count) in enumerate(counts.items()) if count >= min_freq}
    vocab["<UNK>"] = 0
    return vocab

def encode(tokens, vocab):
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]

def decode(indices, vocab):
    inv_vocab = {idx: word for word, idx in vocab.items()}
    return [inv_vocab.get(i, "<UNK>") for i in indices]

def make_training_data(indices, context_size):
    X, y = [], []
    for i in range(context_size, len(indices)):
        X.append(indices[i - context_size:i])
        y.append(indices[i])
    return X, y