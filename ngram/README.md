Here, I trained a trigram (n=3 based n-gram) model that predicts the next word based on the previous two words (hence the tri). It's fully probability based so responses can vary a lot based on the corpus. Here's a few examples of me interacting with this model

### Example 1
```bash
❯ python3 generate.py loops are
Generated: loops are initialized using square brackets
```

### Example 2
```bash
❯ python3 generate.py files can
Generated: files can
```

### Example 3
```bash
❯ python3 generate.py files are
Generated: files are declared using the open function
```

### Example 4
```bash
❯ python3 generate.py conditions allow
Generated: conditions allow the open function
```

### Example 5
```bash
❯ python3 generate.py loops are
Generated: loops are initialized using square brackets
```

These look *okay* compared to ELIZA but it kind of seems like it's memorizing the input, not fully conversing with us.