
## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                               │
│                                                                     │
│  Image.jpg ──→ MobileNetV2 ──→ 1280-d vector (saved to .pkl)       │
│                                                                     │
│  Caption: "a dog runs on grass"                                     │
│  Tokenize: [1, 45, 678, 23, 891, 2]  (start=1, end=2)             │
│                                                                     │
│  Training pairs (teacher forcing):                                  │
│    [feat, [1]]          → predict 45  ("a")                        │
│    [feat, [1,45]]       → predict 678 ("dog")                      │
│    [feat, [1,45,678]]   → predict 23  ("runs")                     │
│    ...                                                              │
│                                                                     │
│  Loss = cross_entropy(predicted_word, true_next_word)              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PHASE                               │
│                                                                     │
│  New Image ──→ MobileNetV2 ──→ feature vector                      │
│                                                                     │
│  Step 1: input=[<start>]           → predict "a"                   │
│  Step 2: input=[<start>, "a"]      → predict "dog"                 │
│  Step 3: input=[<start>,"a","dog"] → predict "is"                  │
│  ...                                                                │
│  Step N: model predicts "<end>"    → STOP                          │
│                                                                     │
│  Output: "a dog is running in the grass"                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Model Architecture (Explained Simply)

```
IMAGE BRANCH                    LANGUAGE BRANCH
──────────────                  ────────────────
Input: 1280-d vector            Input: [word1, word2, ..., wordN]
       │                               │
  Dense(512, relu)               Embedding(256)
       │                               │
   Dropout(0.5)                   Dropout(0.5)
       │                               │
       │                          LSTM(512)
       │                               │
       └──────── ADD (merge) ──────────┘
                      │
               LayerNormalization
                      │
               Dense(512, relu)
                      │
                  Dropout(0.5)
                      │
              Dense(vocab_size)
                      │
                  softmax
                      │
              P(next word | image, previous words)
```

**Why Add and not Concatenate?**
Both work, but `Add` forces the image branch and language branch to live in the **same 512-dimensional space**, which is a stronger constraint that often regularises better on smaller datasets like Flickr8k.

---
