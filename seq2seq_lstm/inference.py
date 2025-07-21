import torch
import sys
from model import EncoderLSTM, DecoderLSTM
from data import eng_vocab, fre_vocab, fre_idx2word

embed_dim = 16
hidden_dim = 32

encoder = EncoderLSTM(len(eng_vocab), embed_dim, hidden_dim)
decoder = DecoderLSTM(len(fre_vocab), embed_dim, hidden_dim)

encoder.load_state_dict(torch.load("model.pth")[0])
decoder.load_state_dict(torch.load("model.pth")[1])

def translate(sentence):
    try:
        input_tensor = torch.tensor([[eng_vocab[word] for word in sentence]])
    except KeyError as e:
        print(f"Word not in vocabulary: {e}")
        return ""

    hidden, cell = encoder(input_tensor)

    decoder_input = torch.tensor([[fre_vocab['<SOS>']]])
    output_words = []

    for _ in range(15):
        output, hidden, cell = decoder(decoder_input, hidden, cell)
        pred_token = output.argmax(dim=1).item()
        if pred_token == fre_vocab['<EOS>']:
            break
        output_words.append(fre_idx2word[pred_token])
        decoder_input = torch.tensor([[pred_token]])

    return ' '.join(output_words)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <word1> <word2> ...")
        exit(1)

    input_words = sys.argv[1:]
    print("Translation:", translate(input_words))