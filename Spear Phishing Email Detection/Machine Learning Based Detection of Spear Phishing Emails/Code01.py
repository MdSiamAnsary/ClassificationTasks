# Importing necessary libraries
import numpy as np        # For numerical operations
import pandas as pd       # For data manipulation and analysis
import matplotlib.pyplot as plt  # For data visualization


# Importing WordCloud for text visualization
from wordcloud import WordCloud

# Importing NLTK for natural language processing
import nltk
from nltk.corpus import stopwords    # For stopwords


# Downloading NLTK data
nltk.download('stopwords')   # Downloading stopwords data
nltk.download('punkt')       # Downloading tokenizer data

# Suppressing Warnings
import warnings
warnings.filterwarnings("ignore")

# Loading the Data
df1 = pd.read_csv('dataset1.csv', encoding='latin1')
print(df1['Category'].value_counts())
df2 = pd.read_csv('dataset2.csv', encoding='latin1')
print(df2['CATEGORY'].value_counts())
df3 = pd.read_csv('dataset3.csv', encoding='latin1')
print(df3['Label'].value_counts())

df3 = df3.drop(columns=['Unnamed: 0'])
df2 = df2.drop(columns=['FILE_NAME'])


df2.rename(columns={"CATEGORY": "Category"}, inplace=True)
df2.rename(columns={"MESSAGE": "Message"}, inplace=True)

df3.rename(columns={"Label": "Category"}, inplace=True)
df3.rename(columns={"Body": "Message"}, inplace=True)

df1=df1.replace({'ham':0,'spam':1})

df4 = pd.DataFrame(columns=['Category', 'Message'], dtype=object)

df4[['Category', 'Message']] = df3[['Category', 'Message']]

frames = [df1, df2, df3]
df = pd.concat(frames)

df.dropna(subset=['Message'], inplace=True)

df = df.drop_duplicates(keep = 'first')

values = df['Category'].value_counts()
print(df['Category'].value_counts())

nMax = values[1]
res = df.groupby('Category').apply(lambda x: x.sample(n=min(nMax, len(x))))
print(res['Category'].value_counts())


# Importing the Porter Stemmer for text stemming
from nltk.stem.porter import PorterStemmer

# Importing the string module for handling special characters
import string

# Creating an instance of the Porter Stemmer
ps = PorterStemmer()

# Lowercase transformation and text preprocessing function
def transform_text(text):
    # Transform the text to lowercase
    text = text.lower()
    
    # Tokenization using NLTK
    text = nltk.word_tokenize(text)
    
    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    # Removing stop words and punctuation
    text = y[:]
    y.clear()
    
    # Loop through the tokens and remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
        
    # Stemming using Porter Stemmer
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    # Join the processed tokens back into a single string
    return " ".join(y)

df['Text'] = df['Message'].apply(transform_text)







