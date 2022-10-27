# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:50:15 2022

@author: brull
"""

import pandas as pd
import json
import gzip
import os
import re
import string


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import en_core_web_sm
spacy.load('en_core_web_sm')


# This is Amanda's attempt and trying some NLP for week 5

# See current working directory
os.getcwd()


# Change working directory to parent directory (where your data is stored)

#os.chdir('C:\\users\\cchee\\OneDrive\\Desktop')

#read the file

#clean_data = pd.read_parquet('team8_initial_clean.parquet.gz', engine ='pyarrow')
clean_data = df

#Check the size of the data

print(clean_data.index)

#Set a seed and sample a random 10,000 rows of data

amanda_sample = clean_data.sample(n=10000, random_state= 54321)

print(amanda_sample.index)

# Expand the contractions

# Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)

# Expanding Contractions in the reviews
amanda_sample['reviewText']=amanda_sample['reviewText'].apply(lambda x:expand_contractions(x))


print(amanda_sample['reviewText'].head(50)) #take a look at a few after expansion

# Lemmatization with stopwords removal

# Loading model 
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

amanda_sample['reviewText']=amanda_sample['reviewText'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))

print(amanda_sample['reviewText'].head(50)) #take a look at a few 

#group our words by main product category

grouped_sample = amanda_sample[['main_cat','reviewText']].groupby(by='main_cat').agg(lambda x:' '.join(x))
print(grouped_sample.head())

# Create a Document Term Matrix - LEFT OFF HERE

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(analyzer='word')
data=cv.fit_transform(grouped_sample['reviewText'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names_out())
df_dtm.index=grouped_sample.index
df_dtm.head(3)