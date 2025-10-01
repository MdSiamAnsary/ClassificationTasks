# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

!pip install pandas numpy scikit-learn nltk spacy textblob transformers vaderSentiment readability-lxml en-core-web-sm

!pip install textstat

!pip install contractions

import pandas as pd
import numpy as np
import string
import re
import nltk
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import ngrams, pos_tag, ne_chunk
from nltk.tree import Tree
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import contractions

import pandas as pd
import numpy as np
import re, string, nltk, spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import pearsonr
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from transformers import BertTokenizer, BertModel
import torch
nltk.download('punkt')
nltk.download('stopwords')

from transformers import BertTokenizer, BertForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_model.eval()

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def extract_features(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    unique_word_count = len(set(words))
    char_count = len(text)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    ttr = unique_word_count / word_count if word_count else 0
    hapax = len([w for w in set(words) if words.count(w) == 1]) / word_count if word_count else 0
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    punctuation_count = sum(1 for c in text if c in string.punctuation)
    stop_word_count = sum(1 for w in words if w.lower() in stopwords.words('english'))
    contraction_count = len([c for c in contractions.fix(text).split() if "'" in c])
    emotion_words = ['happy', 'sad', 'angry', 'fear', 'joy', 'disgust', 'surprise']
    emotion_word_count = sum(1 for w in words if w.lower() in emotion_words)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    vader = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    flesch = textstat.flesch_reading_ease(text)
    fog = textstat.gunning_fog(text)
    first_person_count = sum(1 for w in words if w.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'])
    direct_address_count = sum(1 for w in words if w.lower() in ['you', 'your', 'yours'])
    chunks = ne_chunk(pos_tag(words))
    person_entities = sum(1 for subtree in chunks if isinstance(subtree, Tree) and subtree.label() == 'PERSON')
    date_entities = sum(1 for subtree in chunks if isinstance(subtree, Tree) and subtree.label() == 'DATE')
    bigrams = list(ngrams(words, 2))
    trigrams = list(ngrams(words, 3))
    bigram_uniqueness = len(set(bigrams)) / len(bigrams) if bigrams else 0
    trigram_uniqueness = len(set(trigrams)) / len(trigrams) if trigrams else 0
    syntax_variety = len(set(tag for word, tag in pos_tag(words))) / len(words) if words else 0
    perplexity = calculate_perplexity(text)

    return pd.Series([
        word_count, unique_word_count, char_count, avg_word_length, ttr, hapax, sentence_count, avg_sentence_length,
        punctuation_count, stop_word_count, contraction_count, emotion_word_count, polarity, subjectivity,
        vader, flesch, fog, first_person_count, direct_address_count, person_entities, date_entities,
        bigram_uniqueness, trigram_uniqueness, syntax_variety, perplexity
    ])

feature_names = [
    'WordCount', 'UniqueWordCount', 'CharCount', 'AvgWordLength', 'TTR',
    'HapaxLegomenonRate', 'SentenceCount', 'AvgSentenceLength',
    'PunctuationCount', 'StopWordCount', 'ContractionCount', 'EmotionWordCount',
    'Polarity', 'Subjectivity', 'VaderCompound', 'FleschReadingEase',
    'GunningFog', 'FirstPersonCount', 'DirectAddressCount',
    'PersonEntities', 'DateEntities', 'BigramUniqueness', 'TrigramUniqueness', 'SyntaxVariety', 'Perplexity'
]

df = pd.read_csv("dataset.csv")

df.rename(columns={'generated': 'label', 'text': 'text'}, inplace=True)

df = df.dropna(subset=['text', 'label'])

from sklearn.utils import shuffle
df = shuffle(df)

import nltk
nltk.download('punkt_tab')

import nltk
nltk.download('averaged_perceptron_tagger_eng')

nltk.download('maxent_ne_chunker_tab')

feature_df = df['text'].apply(extract_features)
feature_df.columns = feature_names

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

bert_embeddings = df['text'].apply(get_bert_embedding)
bert_matrix = np.vstack(bert_embeddings.values)

X_full = np.hstack((feature_df.values, bert_matrix))
y = df['label'].values

correlations = [abs(pearsonr(feature_df.iloc[:, i], y)[0]) for i in range(feature_df.shape[1])]
important_indices = [i for i, corr in enumerate(correlations) if corr > 0.1]  # threshold can be tuned
X_reduced = np.hstack((feature_df.iloc[:, important_indices].values, bert_matrix))

X_temp, X_test, y_temp, y_test = train_test_split(X_reduced, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

base_models = [
    ('xgb', XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.03, subsample=0.9,
                          colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ('lgb', LGBMClassifier(n_estimators=300, max_depth=9, learning_rate=0.03, num_leaves=100, subsample=0.9, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3, class_weight='balanced', random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=5, min_samples_split=5, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=300, max_features='sqrt', class_weight='balanced', random_state=42)),
    ('svc', SVC(C=5, kernel='rbf', probability=True, class_weight='balanced', random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance')),
]

meta_model = LogisticRegression(solver='liblinear', penalty='l2', class_weight='balanced')

hybrid_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=5),
    passthrough=True,  
    n_jobs=-1
)

hybrid_model.fit(X_train, y_train)

y_pred = hybrid_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))