import pickle
import random
import sys

def load_model(path="trigram_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def generate_sentence(model, start=None, max_words=20):
    if not start:
        start = random.choice(list(model.keys()))
    
    output = [start[0], start[1]]
    
    for _ in range(max_words - 2):
        key = (output[-2], output[-1])
        next_words = model.get(key)
        if not next_words:
            break
        next_word = random.choice(next_words)
        output.append(next_word)

    return ' '.join(output)

if __name__ == "__main__":
    model = load_model()

    if len(sys.argv) > 2:
        start = (sys.argv[1].lower(), sys.argv[2].lower())
    else:
        start = None

    sentence = generate_sentence(model, start=start)
    print("Generated:", sentence)