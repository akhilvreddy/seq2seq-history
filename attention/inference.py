import torch
import sys
from model import Encoder, AttentionDecoder
from data import eng_vocab, fre_vocab, fre_idx2word
from utils import show_attention

embed_dim = 32
hidden_dim = 64

encoder = Encoder(len(eng_vocab), embed_dim, hidden_dim)
decoder = AttentionDecoder(len(fre_vocab), embed_dim, hidden_dim)

encoder.load_state_dict(torch.load("model.pth")[0])
decoder.load_state_dict(torch.load("model.pth")[1])

def translate(sentence):
    try:
        input_tensor = torch.tensor([[eng_vocab[word] for word in sentence]])
    except KeyError as e:
        print(f"Word not in vocabulary: {e}")
        return "", None

    encoder_outputs, hidden, cell = encoder(input_tensor)

    decoder_input = torch.tensor([[fre_vocab['<SOS>']]])
    output_words = []
    attentions = []

    for _ in range(15):
        output, hidden, cell, attn_weights = decoder(decoder_input, hidden, cell, encoder_outputs)
        pred_token = output.argmax(dim=1).item()
        if pred_token == fre_vocab['<EOS>']:
            break
        output_words.append(fre_idx2word[pred_token])
        attentions.append(attn_weights.squeeze(0).tolist())
        decoder_input = torch.tensor([[pred_token]])

    return output_words, attentions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <word1> <word2> ...")
        exit(1)

    input_words = sys.argv[1:]
    output_words, attention_matrix = translate(input_words)

    if attention_matrix:
        print("Translation:", ' '.join(output_words))
        show_attention(input_words, output_words, attention_matrix)