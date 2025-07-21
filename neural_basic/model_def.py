import torch
import torch.nn as nn

class FFNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embed(x)
        flat = embedded.view(x.size(0), -1)
        out = torch.relu(self.fc1(flat))
        return self.fc2(out)