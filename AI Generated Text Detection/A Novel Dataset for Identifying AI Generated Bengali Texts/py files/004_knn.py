# -*- coding: utf-8 -*-


!pip install bnltk

import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from bnltk.stemmer import BanglaStemmer

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

# Load dataset
data = pd.read_csv('dataset.csv')

# Expanded and detailed Bengali contractions dictionary
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
    "দেখা'হচ্ছে": "দেখা হচ্ছে", "হতে'পারে": "হতে পারে", "করব'না": "করব না",
    "যাচ্ছিলাম'না": "যাচ্ছিলাম না", "বলতাম'না": "বলতাম না", "নিতে'চাই": "নিতে চাই"
}

# Function to expand contractions
def expand_contractions(text):
    for contraction, expansion in bengali_contractions.items():
        text = text.replace(contraction, expansion)
    return text

# Preprocessing function
stemmer = BanglaStemmer()
bengali_punctuations = string.punctuation + "।“”’‘"

def preprocess(text):
    text = str(text).lower()
    text = expand_contractions(text)
    text = re.sub(r'[{}]'.format(re.escape(bengali_punctuations)), '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in bangla_stopwords]
    lemmatized = [stemmer.stem(token) for token in tokens]
    return ' '.join(lemmatized)

import nltk
nltk.download('punkt_tab')

# Apply preprocessing
data['cleaned_text'] = data['text'].apply(preprocess)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['label']

# Train/test split (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# K-Nearest Neighbors classifier
knn_model = KNeighborsClassifier(n_neighbors=15,metric='euclidean',n_jobs=-1)

# Train the model
knn_model.fit(X_train, y_train)

# Predict
y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
