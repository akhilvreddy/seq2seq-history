Just like the blog post explained, here we take our words, embed them, and put them through an encoder/decoder. The hidden state after encoding the last word is what is being used when the decoder starts generated text. 

Here's what I got as a result of running this

```txt
❯ ./inference.sh i am hungry
Testing inference (enter 3 words)
Translation: j' ai faim
```

```txt
❯ ./inference.sh i am sad
Testing inference (enter 3 words)
Translation: je suis triste
```

```txt
❯ ./inference.sh i am happy
Testing inference (enter 3 words)
Translation: je suis content
```

All these sentences are very simple and there's a huge overlap between each of the sentences ("i am" shows up in all), making it easy for the model to predict well.