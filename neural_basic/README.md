Here, we're using randomly initialized embeddings which during training are updated via backpropagation (just like weights in a NN). 

The architecture leanrs to predict the next word and in doing so, it learns useful representations of words. 

## Workflow

Let's say the input is ["the", "cat", "sat"] and we want to predict "on".

1) **Tokenization + Vocabulary**

We first map word to an integer index. 

2) **Embedding Layer**

Input word indices (like `[12, 45, 78]`) are outputted to dense vectors of shape `[3, d_model]`. 

3) **Feedforward Network**

We concatenate the 3 embeddigns (`[3, d_model]`) through either smart ways (with some positional encoding) or just plain batch averaging. We then feed this through 1 or 2 linear layers → ReLU → Linear → Softmax (very similar to vanilla word2vec).

4) **Loss**

Cross-Entropy loss is used because it lets us compare the predicted distribution to the index of "on".

---

## Results

```txt
❯ python inference.py the cat # n = 6  
the cat sat on the mat and the
```

It's cool to see that this generated something kind of coherent. The end of the sentence started to drift but it was cool to see words that were initially close to each other start to show up in a similar fashion. 