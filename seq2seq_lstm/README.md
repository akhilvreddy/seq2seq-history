I couldn't really show the usefulness of the LSTM architecture without using a huge corpus. But from what we can see, LSTMs can easily handle longer windows.

```txt
❯ ./inference.sh i am very hungry
Translating: i am very hungry
Translation: j' ai très faim
```

```txt
❯ ./inference.sh she is not at home
Translating: she is not at home
Translation: elle n' est pas à la maison
```