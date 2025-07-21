import re
from collections import defaultdict
import pickle

def tokenize(text):
    return re.findall(r"\b\w+(?:[().`']\w+)*\b", text)

def build_trigram_model(file_path):
    model = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            tokens = tokenize(line.lower())
            if len(tokens) < 3:
                continue
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i+1])
                next_word = tokens[i+2]
                model[key].append(next_word)

    return model

def save_model(model, out_path="trigram_model.pkl"):
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    corpus_path = "corpus.txt"
    model = build_trigram_model(corpus_path)
    save_model(model)
    print(f"Trigram model saved with {len(model)} entries.")