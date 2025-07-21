import torch
import pickle
from utils import tokenize, encode
from model_def import FFNLM
import sys

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

context_size = 2
embed_dim = 10
hidden_dim = 64
vocab_size = len(vocab)

model = FFNLM(vocab_size, embed_dim, context_size, hidden_dim)
model.load_state_dict(torch.load("ffnlm.pth"))
model.eval()

def predict_next_word(context_words):
    tokens = tokenize(" ".join(context_words))
    indices = encode(tokens, vocab)

    if len(indices) < context_size:
        indices = [vocab["<UNK>"]] * (context_size - len(indices)) + indices
    elif len(indices) > context_size:
        indices = indices[-context_size:]

    input_tensor = torch.tensor([indices])

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_idx = torch.argmax(logits, dim=-1).item()

    inv_vocab = {idx: word for word, idx in vocab.items()}
    return inv_vocab.get(predicted_idx, "<UNK>")

def generate_sequence(seed, length=10):
    output = seed[:]
    for _ in range(length):
        next_word = predict_next_word(output[-context_size:])
        output.append(next_word)
    return " ".join(output)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage is: python inference.py <word1> <word2>")
        sys.exit(1)
    
    seed = [sys.argv[1], sys.argv[2]]
    print(generate_sequence(seed, length=6))