eng_sentences = [
    ['i', 'am', 'hungry'],
    ['i', 'am', 'sad'],
    ['i', 'am', 'happy']
]

fre_sentences = [
    ["j'", 'ai', 'faim'],
    ['je', 'suis', 'triste'],
    ['je', 'suis', 'content']
]

def build_vocab(sentences, special_tokens=['<PAD>', '<SOS>', '<EOS>']):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

eng_vocab = build_vocab(eng_sentences)
fre_vocab = build_vocab(fre_sentences)

eng_idx2word = {idx: word for word, idx in eng_vocab.items()}
fre_idx2word = {idx: word for word, idx in fre_vocab.items()}

def get_data():
    input_tensors = []
    target_tensors = []
    for eng, fre in zip(eng_sentences, fre_sentences):
        input_tensor = [eng_vocab[word] for word in eng]
        target_tensor = [fre_vocab['<SOS>']] + [fre_vocab[word] for word in fre] + [fre_vocab['<EOS>']]
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)
    return input_tensors, target_tensors