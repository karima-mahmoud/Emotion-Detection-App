# 🧠 Emotion Detection App (PyTorch + Gradio)

This project is an **Emotion Detection System** that uses NLP preprocessing and a deep learning model to classify English text into one of the six emotions:
**Joy, Sadness, Love, Anger, Fear, Surprise**.

Built with:
- **PyTorch** for training the neural network
- **Scikit-learn** for TF-IDF vectorization
- **Gradio** for building the interactive user interface

---

## 📊 Dataset

The dataset used is split into:
- `emotion_train.csv`
- `emotion_validation.csv`
- `emotion_test.csv`

Each file contains:
- `text`: the input sentence
- `label`: the emotion label (encoded as integers)

---

## 🧹 Preprocessing Steps

Text data undergoes the following steps:
1. Remove usernames (`@user`)
2. Convert to lowercase
3. Convert emojis to text using `emoji.demojize`
4. Remove punctuation
5. Remove stopwords
6. Lemmatization using POS tagging
7. Remove numbers and single characters
8. Remove unwanted words (e.g. "lol", "gonna", etc.)

---

## 🧠 Model Architecture

The neural network has the following layers:

- `Linear(input_size → 256)`
- `BatchNorm1d`
- `Dropout(0.5)`
- `Linear(256 → 128)`
- `BatchNorm1d`
- `Dropout(0.3)`
- `Linear(128 → 6)` → Final output logits

Trained using:
- CrossEntropyLoss
- Adam optimizer
- TF-IDF vectorized inputs

---

## 📈 Training & Evaluation

Training is performed over 15 epochs and validated on a separate set. The best model (with highest validation accuracy) is saved as `best_model.pth`.

Performance is visualized using matplotlib for:
- Loss per epoch
- Accuracy per epoch

---

## 🖥️ Gradio Interface

The app includes a web interface using **Gradio**:

```bash
# Run the app locally
python app.py

---
## 🙋‍♂️ Author

**👨‍💻 Karima Mahmoud**  
📫 karimamahmoudsalem1@gmail.com  
🐙 GitHub: https://github.com/karima-mahmoud




