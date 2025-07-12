# -*- coding: utf-8 -*-


!pip install transformers sentencepiece nltk bnltk tensorflow scikit-learn

import pandas as pd
import numpy as np
import re
import string
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, GlobalMaxPooling1D, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel
from nltk.tokenize import word_tokenize
from bnltk.stemmer import BanglaStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

import nltk
nltk.download('punkt_tab')

# Load dataset
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

# Encode labels
labels = sorted(df['label'].unique().tolist())
label_map = {label: idx for idx, label in enumerate(labels)}
df['label_enc'] = df['label'].map(label_map)

# Tokenization with BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
MAX_LEN = 128

input_ids, attention_masks = [], []
for text in tqdm(df['cleaned_text']):
    encodings = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    input_ids.append(encodings['input_ids'][0])
    attention_masks.append(encodings['attention_mask'][0])

X_input_ids = np.array(input_ids)
X_attention_masks = np.array(attention_masks)
y = to_categorical(df['label_enc'], num_classes=len(labels))

# Train/test split
X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
    X_input_ids, X_attention_masks, y, test_size=0.3, random_state=42, stratify=y
)

from transformers import AutoTokenizer, TFBertModel
import tensorflow as tf

# Load BERT
bert = TFAutoModel.from_pretrained("sagorsarker/bangla-bert-base")

from tensorflow.keras.layers import Lambda

def get_bert_output(inputs):
    input_ids, attention_mask = inputs
    return bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

# Model architecture: CNN-LSTM ensemble over BERT
input_ids_layer = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
attention_mask_layer = Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

# Wrap BERT in Lambda with output shape specified
def get_bert_output(inputs):
    input_ids, attention_mask = inputs
    return bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

bert_output = Lambda(get_bert_output, output_shape=(MAX_LEN, 768))([input_ids_layer, attention_mask_layer])

# Ensemble: CNN + LSTM
conv = Conv1D(filters=128, kernel_size=3, padding='same')(bert_output)
act = LeakyReLU()(conv)
pool = GlobalMaxPooling1D()(act)


pool_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(pool)

lstm_out = LSTM(64)(pool_expanded)
dense = Dense(64)(lstm_out)
dense = LeakyReLU()(dense)
dropout = Dropout(0.3)(dense)
output = Dense(len(labels), activation='softmax')(dropout)

model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)
model.summary()

model.compile(optimizer=Adam(learning_rate=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    [X_train_ids, X_train_masks],
    y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=16,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

# Predict
y_pred_probs = model.predict([X_test_ids, X_test_masks])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Evaluation
print("Evaluation Metrics:")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
