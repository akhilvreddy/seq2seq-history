import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dense_language_model import model, vocab

with torch.no_grad():
    embeddings = model.embed.weight.detach().numpy() # shape: [vocab_size, embed_dim]

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# inverse vocab (index to word)
inv_vocab = {idx: word for word, idx in vocab.items()}

plt.figure(figsize=(8, 6))
for idx, (x, y) in enumerate(reduced):
    word = inv_vocab.get(idx, "<UNK>")
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word)
plt.title("Learned Word Embeddings")
plt.grid(True)
plt.show()