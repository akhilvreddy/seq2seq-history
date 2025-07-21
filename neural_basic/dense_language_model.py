import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from utils import tokenize, build_vocab, encode, make_training_data
from model_def import FFNLM

with open("corpus.txt", "r") as f:
    corpus = f.read()

tokens = tokenize(corpus)
vocab = build_vocab(tokens)
if "<UNK>" not in vocab:
    vocab["<UNK>"] = len(vocab)
indices = encode(tokens, vocab)

context_size = 2
X_data, y_data = make_training_data(indices, context_size)
X = torch.tensor(X_data)
y = torch.tensor(y_data)

vocab_size = len(vocab)
embed_dim = 10
hidden_dim = 64
epochs = 100

model = FFNLM(vocab_size, embed_dim, context_size, hidden_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    logits = model(X)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "ffnlm.pth")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)