import torch
import torch.nn as nn
import torch.optim as optim
from model import Encoder, AttentionDecoder
from data import get_data, eng_vocab, fre_vocab

embed_dim = 32
hidden_dim = 64

encoder = Encoder(len(eng_vocab), embed_dim, hidden_dim)
decoder = AttentionDecoder(len(fre_vocab), embed_dim, hidden_dim)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.005)

input_tensors, target_tensors = get_data()

for epoch in range(800):
    total_loss = 0
    for inp, tgt in zip(input_tensors, target_tensors):
        input_tensor = torch.tensor([inp])
        target_tensor = torch.tensor([tgt])

        encoder_outputs, hidden, cell = encoder(input_tensor)

        decoder_input = torch.tensor([[fre_vocab['<SOS>']]])
        loss = 0

        for t in range(1, target_tensor.size(1)):
            output, hidden, cell, _ = decoder(decoder_input, hidden, cell, encoder_outputs)
            loss += loss_fn(output, target_tensor[:, t])
            decoder_input = target_tensor[:, t].unsqueeze(1) # teacher forcing

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save((encoder.state_dict(), decoder.state_dict()), "model.pth")