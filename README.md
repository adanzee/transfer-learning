#  Image Captioning using CNN + RNN (LSTM)
### A Deep Learning Lab — Flickr8k Dataset

> *Teaching a machine to look at a photo and describe it in plain English — one word at a time.*


## 🧠 Project Overview

This project builds an **Image Captioning system** — a model that takes any image as input and automatically generates a human-readable sentence describing what it sees. It is a classic **multimodal AI** problem, meaning it bridges two completely different data types: **images** and **text**.

The approach uses two neural networks working together:

| Network | Role |
|---------|------|
| **CNN (MobileNetV2)** | "See" the image — extract what objects, colors, and context exist |
| **RNN (LSTM)** | "Say" what was seen — generate a sequence of words describing it |

**Example:**
```
Input  → 🖼️  [photo of a brown dog running through green grass]
Output → "a brown dog is running through a grassy field"
```

---

## 📚 What I Learned

This lab was not just about writing code. It forced me to deeply understand how modern AI systems actually *think about* images and *speak* about them. Here is what I genuinely learned:

### 1. **Transfer Learning is Incredibly Powerful**
I did not train MobileNetV2 from scratch (which would require millions of images and weeks of compute). Instead, I used a model **already trained on 1.2 million ImageNet images**, stripped off its classification head, and used its internal feature extractor. The model already knows what a "dog", "tree", or "water" looks like. I just reused that knowledge.

> 💡 *Lesson: You almost never train a CNN from scratch in real projects. Transfer learning is the standard.*

### 2. **Images and Text Live in Completely Different Worlds**
An image is a grid of pixel numbers. A word is a token in a sequence. To make a model that understands both, you have to **project them into the same mathematical space** — a shared vector representation where both images and words can be compared and combined. This is what the `Dense(512)` and `Embedding(256)` layers do.

### 3. **LSTMs Remember Context Across Time**
Before this project, I understood the *idea* of RNNs. After this project, I understand *why* they matter for language. When the model generates the word "running", its LSTM hidden state remembers that it already said "dog is". This memory is what makes the next prediction ("through") coherent. Without sequential memory, each word would be predicted in isolation and make no sense.

### 4. **Teacher Forcing Stabilises RNN Training**
During training, instead of feeding the model's *own predicted word* back as input (which is what happens at test time), we feed the **correct ground-truth word**. This prevents one bad prediction from cascading into completely wrong future predictions during early training. It's a clever trick that makes training much faster and more stable.

### 5. **BLEU Score — How to Evaluate Language Quality**
Accuracy alone cannot tell you if a caption is good. "A dog runs" and "A canine is sprinting" mean the same thing but share few words. BLEU (Bilingual Evaluation Understudy) measures how much **n-gram overlap** exists between a generated caption and reference captions. I learned to compute BLEU-1 through BLEU-4 and interpret what they mean.

### 6. **Greedy Decoding vs. Beam Search**
At inference time, the model predicts a probability distribution over the vocabulary at each step. The simplest approach — **greedy decoding** — just picks the highest-probability word. I learned that this can get "stuck" in suboptimal sequences, and that **beam search** (keeping the top-k partial sequences) produces better captions at the cost of more computation.

### 7. **Data Generators Are Essential for Large Datasets**
Loading all 8,000 images into RAM simultaneously would crash most machines. I learned to write a **Python generator** that yields one batch at a time, allowing training on datasets far larger than available memory.

---

## 📦 Dataset

| Property | Detail |
|----------|--------|
| Name | **Flickr8k** |
| Size | 8,000 images |
| Captions per image | 5 human-written captions |
| Total captions | ~40,000 |
| Image format | JPEG |
| Source | [Kaggle — Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) |
| Train / Val / Test | 6,400 / 800 / 800 images |


#

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install tensorflow pillow matplotlib tqdm numpy nltk
```

### Download the Dataset
1. Go to [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. Download and unzip
3. You should have:
   - `flickr8k/Images/` — folder with .jpg files
   - `flickr8k/captions.txt`

### Set Paths
Open `image_captioning_rnn.ipynb` (or `.py`) and edit **Cell 1**:
```python
IMAGES_DIR    = './flickr8k/Images'       # ← your path here
CAPTIONS_FILE = './flickr8k/captions.txt' # ← your path here
```

---

## ▶️ How to Run

### Option A — Jupyter Notebook (Recommended)
```bash
jupyter notebook image_captioning_rnn.ipynb
```
Run cells one by one. The notebook has detailed markdown explanations between each step.

### Option B — Python Script
```bash
python image_captioning_rnn.py
```

### ⏱️ Expected Runtime

| Step | Time (CPU) | Time (GPU) |
|------|-----------|-----------|
| Feature extraction (first run) | ~15–25 min | ~3–5 min |
| Feature extraction (cached) | ~5 sec | ~5 sec |
| Training (20 epochs) | ~2–4 hours | ~20–40 min |
| Hyperparameter experiments | ~30 min | ~5 min |

> **Tip:** Feature extraction only runs once. After that, features are loaded from `features_mobilenet.pkl` instantly.

---

## 📊 Results & Evaluation

### Metrics Used

| Metric | What It Measures |
|--------|-----------------|
| **Word-Level Accuracy** | % of next-word predictions correct (training signal) |
| **Loss** | Categorical cross-entropy — how uncertain the model is |
| **BLEU-1** | Unigram word overlap between generated and reference captions |
| **BLEU-2** | Bigram (2-word phrase) overlap |
| **BLEU-3** | Trigram overlap |
| **BLEU-4** | 4-gram overlap — strictest, most standard metric |

### Typical Results on Flickr8k

| Split | Accuracy | Loss |
|-------|----------|------|
| Train | ~0.55–0.65 | ~2.5 |
| Val   | ~0.45–0.55 | ~2.9 |
| Test  | ~0.44–0.54 | ~3.0 |

| BLEU Score | Typical Value |
|------------|--------------|
| BLEU-1 | 0.55 – 0.62 |
| BLEU-2 | 0.35 – 0.42 |
| BLEU-3 | 0.24 – 0.30 |
| BLEU-4 | 0.15 – 0.22 |

> A train accuracy of ~60% does not mean the model is "bad". Predicting the exact next word out of a 5,000-word vocabulary at every step is genuinely hard — and the captions still come out readable and contextually correct.

### Output Plots Generated
- **`training_curves_(Baseline).png`** — Loss and accuracy over epochs for both train and val sets
- **`captioning_results.png`** — 6 test images side-by-side with generated vs. reference captions
- **`hyperparameter_comparison.png`** — Grouped bar chart comparing 4 configurations

---

## 🔬 Hyperparameter Experiments

The notebook runs 4 configurations and compares them:

| Config | LSTM Units | Batch Size | Observation |
|--------|-----------|------------|-------------|
| **Small** | 256 | 32 | Trains fastest, fewer parameters, may underfit on complex scenes |
| **Base** (default) | 512 | 64 | Best balance of speed and quality |
| **Large** | 1024 | 32 | Most expressive, but risks overfitting on Flickr8k's 6,400 training images |
| **Big Batch** | 512 | 128 | Fastest wall-clock time per epoch; gradient estimates are noisier |

**What changing epochs does:**
- Too few → model hasn't learned enough; captions are generic ("a man is standing")
- Just right → model describes specific content ("a brown dog jumps over a hurdle")
- Too many → model overfits; val/test accuracy drops even as train accuracy climbs

---

## ⚠️ Challenges I Faced

### 1. Caption File Format Differences
The Flickr8k dataset comes in two slightly different caption formats depending on the download source. I had to write a parser that handles both the tab-separated `Flickr8k.token.txt` format and the comma-separated `captions.txt` format without breaking.

### 2. Memory Management
Storing all image features in RAM (about 8,000 × 1,280 floats) takes ~80MB — manageable. But naively loading all pixel data for every batch would spike to several GB. Using the generator pattern solved this.

### 3. Training Instability Early On
In early runs, the loss would spike wildly after epoch 1. The fix was adding `ReduceLROnPlateau` and `LayerNormalization`. The normalisation layer was the biggest single improvement in training stability.

### 4. `<start>` and `<end>` Token Alignment
At inference time, the model must be seeded with the `<start>` token and must stop exactly when it predicts `<end>`. Getting the tokenizer IDs right for these special tokens, and making sure they were in the training vocabulary, required careful debugging.

### 5. BLEU Score Requires NLTK
`corpus_bleu` from `nltk.translate.bleu_score` requires the references to be lists of lists of strings. Getting the shape of the reference input right (one image → multiple reference captions → each caption as a list of words) took several iterations to get correct.

---

## 💡 Key Takeaways

> These are the ideas I would explain to someone who asked me "what did you learn from this project?"

1. **CNNs and RNNs can be combined** — they do not have to work independently. In this project, a frozen CNN feeds a trainable RNN, and they cooperate to produce language from images.

2. **The `<start>` token is not a trick — it is essential.** The LSTM needs an initial hidden state AND an initial input. `<start>` is that first input. Without it, there is no defined starting point for generation.

3. **Pretrained models are not magic.** MobileNetV2 was trained to classify images, not to generate captions. It does not "understand" images the way humans do. It compresses images into a vector that contains enough information for a downstream model to do useful things with.

4. **Word-level accuracy is misleading.** 55% accuracy sounds poor, but each step is predicting 1 of 5,000 possible words — random chance would give 0.02%. The captions the model generates are often readable and contextually appropriate.

5. **Overfitting is a real danger on small datasets.** Flickr8k has only 6,400 training images. The model can easily memorise training captions instead of learning to generalise. Dropout, early stopping, and a moderate vocabulary size all help prevent this.

6. **Evaluation of generative models is fundamentally hard.** BLEU is imperfect — two captions can be semantically identical but have low BLEU overlap. This is an active research problem (newer metrics like CIDEr and SPICE exist specifically for image captioning).

---

## 🚀 Future Improvements

Ideas to extend this project beyond the lab requirements:

- **Beam Search Decoding** — instead of greedy argmax, keep top-k sequences at each step. Usually improves BLEU by 2–5 points.

- **Attention Mechanism** — instead of using a single global image feature vector, use a spatial feature map (7×7×1280) and let the model *attend* to different regions of the image for each word. This is how modern captioning systems work and dramatically improves quality.

- **Larger Dataset** — train on MSCOCO (120,000 images) for significantly better captions.

- **Transformer Decoder** — replace the LSTM with a Transformer decoder block (multi-head self-attention + cross-attention). This is the architecture behind BLIP and CLIP-based captioners.

- **Flask / Streamlit Demo** — wrap the trained model in a simple web app where you can upload any image and get a caption in real time.

- **Caption Quality Visualisation** — highlight which regions of the image the model "looked at" for each word (requires attention maps).

---

## 👨‍💻 Author Notes

This project was built as part of a **Deep Learning lab assignment** using the Flickr8k dataset. It is designed to be educational — every design choice (merge architecture, teacher forcing, feature caching, BLEU evaluation) is explained not just *implemented*. The goal was to understand image captioning from the ground up, not just to copy a working model.

The most valuable part of this lab was not getting the BLEU scores to go up. It was understanding **why** each piece of the pipeline is there — what breaks if you remove the `<start>` token, what happens if you use the test set for tokenizer fitting, why LayerNorm helps where BatchNorm hurts. That understanding is what transfers to the next project.

---

## 📖 References & Further Reading

- **Vinyals et al. (2015)** — *Show and Tell: A Neural Image Caption Generator* — the original merge-architecture paper
- **Bahdanau et al. (2015)** — *Neural Machine Translation by Jointly Learning to Align and Translate* — the attention mechanism this could be extended with
- **MobileNetV2 paper** — Sandler et al. (2018) — the CNN backbone used here
- **BLEU score** — Papineni et al. (2002) — the evaluation metric used
- [Keras Image Captioning Tutorial](https://keras.io/examples/vision/image_captioning/)
- [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

*Built with TensorFlow/Keras · MobileNetV2 · LSTM · Flickr8k*
