# -*- coding: utf-8 -*-


pip install transformers datasets nltk bnltk scikit-learn torch

import pandas as pd
import re
import string
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from bnltk.stemmer import BanglaStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import nltk
nltk.download('punkt_tab')

df = pd.read_csv("dataset.csv")

# Expanded Bengali contraction dictionary
bengali_contractions = {
    "তুমি'র": "তোমার", "আমি'র": "আমার", "সে'র": "তার", "ওর'টা": "ওটার", "এটা'র": "এইটার",
    "হয়নি": "হয় নি", "করছো'না": "করছো না", "বলছি'না": "বলছি না", "যাচ্ছি'না": "যাচ্ছি না",
    "দেখে'ছি": "দেখেছি", "যাবে'না": "যাবে না", "হবে'না": "হবে না", "নেই'তো": "নেই তো",
    "নেই'না": "নেই না", "দেবো'না": "দেবো না", "হয়'তো": "হয় তো", "হয়েছে'না": "হয়েছে না",
    "পারবো'না": "পারবো না", "করবো'না": "করবো না", "জানতাম'না": "জানতাম না", "চাই'না": "চাই না",
    "নিচ্ছি'না": "নিচ্ছি না", "খাই'নি": "খাই নি", "দেখছি'না": "দেখছি না", "দিচ্ছি'না": "দিচ্ছি না",
    "নিয়েছি'না": "নিয়েছি না", "পড়ছি'না": "পড়ছি না", "জানিনা": "জানি না", "বোঝো'না": "বোঝো না",
    "হাসো'না": "হাসো না", "চলো'না": "চলো না", "আসো'না": "আসো না", "যাই'না": "যাই না",
    "চাই'ছিলাম": "চাইছিলাম", "কর'তে": "করতে", "হচ্ছে'না": "হচ্ছে না", "দেখা'ই": "দেখাই",
    "তুমি'ই": "তুমিই", "আমি'ই": "আমিই", "সে'ই": "সেই", "তাদের'টা": "তাদেরটা",
    "আমাদের'টা": "আমাদেরটা", "তোমাদের'টা": "তোমাদেরটা", "করেছি'না": "করেছি না",
    "বলেছি'না": "বলেছি না", "দিয়েছি'না": "দিয়েছি না", "খেয়েছি'না": "খেয়েছি না",
    "নেবে'না": "নেবে না", "যেতে'চাই": "যেতে চাই", "থাকবো'না": "থাকবো না",
    "দেখা'হচ্ছে": "দেখা হচ্ছে", "হতে'পারে": "হতে পারে", "করব'না": "করব না"
}

bangla_stopwords = [
    'অথবা', 'অনুযায়ী', 'অতএব', 'অন্য', 'অবশ্য', 'অবধি', 'অধীন', 'অথচ', 'অর্থাৎ',
    'অনেক', 'অনেকে', 'অন্তত', 'আজ', 'আগে', 'আগামী', 'আছে', 'আছেন', 'আবার', 'আদি',
    'ইহা', 'ইহাতে', 'ইত্যাদি', 'ইনিয়ে', 'এই', 'এখন', 'এখানে', 'এত', 'এবং', 'এটি',
    'এটা', 'এরা', 'এবার', 'এক', 'একই', 'একটা', 'একজন', 'একটু', 'একাধিক', 'একে',
    'এখনো', 'একেবারে', 'এদের', 'এদেরকে', 'এদেরও', 'এসব', 'এসো', 'এসেছে', 'এসেই',
    'ঐ', 'ও', 'ওই', 'ওরা', 'ওদের', 'ওখানে', 'ওদিকে', 'ওর', 'ওইটা', 'ওটা', 'ওদেরকে',
    'ওখানেই', 'কখনো', 'কত', 'কবে', 'কখন', 'কোন', 'কোনও', 'কোনো', 'কোনদিকে', 'কোনটা',
    'কারণ', 'কারও', 'কারো', 'কি', 'কিন্তু', 'কিছু', 'কিছুই', 'কী', 'কিরকম', 'কেবল',
    'কে', 'কেউ', 'কেন', 'কেননা', 'কেই', 'কোথা', 'কোথাও', 'কোথায়', 'খুব', 'গিয়েছিল',
    'গিয়েছে', 'গেছে', 'চলে', 'চান', 'চাই', 'চেয়ে', 'চালু', 'চালানো', 'ছাড়া', 'ছাড়াও',
    'ছিল', 'ছিলাম', 'ছিলেন', 'ছিলে', 'জানেন', 'জানানো', 'জায়গায়', 'জায়গা', 'জানিয়ে',
    'জানি', 'জানতে', 'তবে', 'তবুও', 'তাহলে', 'তাদের', 'তাদেরকে', 'তাহার', 'তারা', 'তাও',
    'তাদেরই', 'তাঁর', 'তাঁরা', 'তুমি', 'তোর', 'তোদের', 'তোকে', 'তোমরা', 'তোমাকে',
    'তাঁদের', 'থাকা', 'থাকে', 'থাকেন', 'থাকলে', 'থাকবেন', 'থাকায়', 'থাকতে', 'থাকছে',
    'দিয়ে', 'দেয়', 'দেন', 'দিতে', 'দেখা', 'দেখে', 'দেখেন', 'দেখানো', 'দেয়া', 'দিলেন',
    'দিলাম', 'দিল', 'দিয়েছে', 'দিচ্ছে', 'ধরনের', 'ধরনে', 'নয়', 'না', 'নাকি', 'নেই',
    'নেওয়া', 'নেওয়ায়', 'নিতে', 'নিজেই', 'নিজে', 'নিজের', 'নিয়ে', 'নিয়েই', 'নিয়েও',
    'নিচে', 'নিবে', 'নিতেও', 'নিতে হবে', 'নিতে হয়', 'পরে', 'পরেই', 'পারে', 'পারেন',
    'পারেনি', 'পারলে', 'পারি', 'পারেনা', 'পাওয়া', 'পেয়েছে', 'পাই', 'প্রতি', 'প্রতিটি',
    'প্রথম', 'প্রভৃতি', 'ফলে', 'বার', 'ব্যাপারে', 'বলে', 'বলেছে', 'বললেন', 'বললেনও',
    'বলা', 'বলতে', 'বলল', 'বলেন', 'বলেছিল', 'বসে', 'বসে পড়া', 'বসে আছে', 'বসে থাকুন',
    'বারে', 'বিনা', 'বিশেষ', 'বেশ', 'বেশি', 'ভালো', 'মধ্যে', 'মধ্যে দিয়ে', 'মধ্যেও',
    'মোটেও', 'মতো', 'মতোই', 'মাঝে', 'মানুষ', 'মনে', 'মনে হয়', 'মনে রাখতে',
    'মনে হয়েছিল', 'যথেষ্ট', 'যদি', 'যদিও', 'যা', 'যাতে', 'যার', 'যারা', 'যাকে',
    'যেখানে', 'যত', 'যখন', 'যারাও', 'যেন', 'যিনি', 'যেহেতু', 'যেতে', 'যাও', 'যাচ্ছে',
    'যায়', 'যাবে', 'যাবেন', 'যেতে হবে', 'যেতে হয়', 'রাখা', 'রাখতে', 'রাখবে', 'রাখেন',
    'রয়েছে', 'রেখে', 'লাগে', 'লাগতে', 'লাগে না', 'লাগবে', 'লাগতে পারে', 'লাগলে',
    'লাগেনি', 'লাগছে', 'লিখে', 'লিখতে', 'লিখলেন', 'শুধু', 'সব', 'সবাই', 'সবচেয়ে',
    'সবচাই'
]

# Preprocess functions
stemmer = BanglaStemmer()
punctuations = string.punctuation + "।“”’‘"

def expand_contractions(text):
    for c, e in bengali_contractions.items():
        text = text.replace(c, e)
    return text

def preprocess(text):
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(f"[{re.escape(punctuations)}]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in bangla_stopwords]
    stemmed = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed)

df['cleaned_text'] = df['text'].astype(str).apply(preprocess)

# Label encoding
labels = df['label'].unique().tolist()
label2id = {label: idx for idx, label in enumerate(labels)}
df['label_id'] = df['label'].map(label2id)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
MAX_LEN = 512

class BengaliDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding='max_length', truncation=True,
                                   max_length=MAX_LEN, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['cleaned_text'], df['label_id'], test_size=0.3, stratify=df['label_id'], random_state=42)

train_dataset = BengaliDataset(train_texts.tolist(), train_labels.tolist())
test_dataset = BengaliDataset(test_texts.tolist(), test_labels.tolist())

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    "sagorsarker/bangla-bert-base",
    num_labels=len(labels)
).to(device)

# Optimizer & scheduler
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.03)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

num_training_steps = len(train_loader) * 10  # 10 epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training
model.train()
for epoch in range(10):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
preds, trues = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())

# Metrics
print("Evaluation Metrics:")
print(f"Accuracy : {accuracy_score(trues, preds):.4f}")
print(f"Precision: {precision_score(trues, preds, average='macro', zero_division=0):.4f}")
print(f"Recall   : {recall_score(trues, preds, average='macro', zero_division=0):.4f}")
print(f"F1 Score : {f1_score(trues, preds, average='macro', zero_division=0):.4f}")
