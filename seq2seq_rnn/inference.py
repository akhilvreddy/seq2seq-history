import torch
from model import Encoder, Decoder
from data import eng_vocab, fre_idx2word, fre_vocab

embed_dim = 8
hidden_dim = 16

encoder = Encoder(len(eng_vocab), embed_dim, hidden_dim)
decoder = Decoder(len(fre_vocab), embed_dim, hidden_dim)

encoder.load_state_dict(torch.load("model.pth")[0])
decoder.load_state_dict(torch.load("model.pth")[1])

def translate(sentence):
    input_tensor = torch.tensor([[eng_vocab[word] for word in sentence]])
    encoder_hidden = encoder(input_tensor)

    decoder_input = torch.tensor([[fre_vocab['<SOS>']]])
    decoder_hidden = encoder_hidden

    output_words = []
    for _ in range(10):
        output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        pred_token = output.argmax(dim=1).item()
        if pred_token == fre_vocab['<EOS>']:
            break
        output_words.append(fre_idx2word[pred_token])
        decoder_input = torch.tensor([[pred_token]])

    return ' '.join(output_words)

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <word1> <word2> <word3>")
        exit(1)

    input_words = sys.argv[1:]
    print("Translation:", translate(input_words))