import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x) # (batch, seq_len, embed_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell # we need all hidden states for attention

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.size()

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1) # (batch, seq_len, hidden)

        concat = torch.cat((decoder_hidden, encoder_outputs), dim=2) # (batch, seq_len, 2*hidden)
        energy = torch.tanh(self.attn(concat)) # (batch, seq_len, hidden)
        scores = self.v(energy).squeeze(2) # (batch, seq_len)
        attn_weights = torch.softmax(scores, dim=1) # (batch, seq_len)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs) # (batch, 1, hidden)
        return context.squeeze(1), attn_weights # (batch, hidden), (batch, seq_len)

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = AdditiveAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_step) # (batch, 1, embed_dim)
        context, attn_weights = self.attention(hidden[-1], encoder_outputs) # context: (batch, hidden)
        rnn_input = torch.cat((embedded.squeeze(1), context), dim=1).unsqueeze(1) # (batch, 1, embed+hidden)

        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell)) # (batch, 1, hidden)
        combined = torch.cat((output.squeeze(1), context), dim=1) # (batch, hidden*2)
        logits = self.fc(combined) # (batch, vocab_size)
        return logits, hidden, cell, attn_weights