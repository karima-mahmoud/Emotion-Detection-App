import gradio as gr
import torch
import torch.nn.functional as F
import pickle
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from emoji import demojize
import neattext.functions as nfx
import string
import re
import numpy as np

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Emotion labels (based on your dataset)
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Define your model architecture
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load trained model
input_size = vectorizer.transform(["sample"]).shape[1]
model = EmotionClassifier(input_size=input_size, num_classes=6)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
model.eval()

# Preprocessing function
def preprocess_text(text):
    text = nfx.remove_userhandles(str(text))
    text = text.lower()
    text = demojize(text)
    text = text.translate(str.maketrans('', '', string.punctuation + '“”’‘'))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        text = ' '.join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tags])
    except:
        pass
    
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction function
def predict_emotion(text):
    cleaned = preprocess_text(text)
    vector = vectorizer.transform([cleaned])
    tensor = torch.FloatTensor(vector.toarray())
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).numpy()[0]
        pred = np.argmax(probs)
    return {emotion_labels[i]: float(probs[i]) for i in range(len(probs))}

# Gradio interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=3, placeholder="Enter a sentence..."),
    outputs=gr.Label(num_top_classes=3),
    title="Emotion Detection App",
    description="Enter a sentence and the model will predict the emotion (joy, sadness, anger, etc.)."
)

# Launch
if __name__ == "__main__":
    interface.launch()
