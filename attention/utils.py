import matplotlib.pyplot as plt
import seaborn as sns

def show_attention(input_tokens, output_tokens, attn_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_matrix, xticklabels=input_tokens, yticklabels=output_tokens, cmap='viridis', linewidths=0.5, annot=True, fmt=".2f")
    plt.xlabel("Input (English)")
    plt.ylabel("Output (French)")
    plt.title("Attention Map")
    plt.tight_layout()
    plt.show()