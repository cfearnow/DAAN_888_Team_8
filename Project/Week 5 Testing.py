# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:22:28 2022

@author: brull
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from datetime import datetime
import json
import gzip
import os
import codecs
import csv
#import boto3
import io
import re
import string
import glob

# Packages for NLP
import spacy
from spacy.tokens import Span
from spacy import displacy

# # Change working directory to parent directory (where your data is stored)
# try:
#     if str.split(os.getcwd(),"\\")[2] == "brull":
#         os.chdir('C:\\Users\\brull\\OneDrive - The Pennsylvania State University\\Team-8\\Data\\clean')
#     else:
#         os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data\\clean')
# except:
#     os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data\\clean')

# # Load cleaned file
# file = 'team8_initial_clean.parquet.gz'
# df = pd.read_parquet(file, engine = "pyarrow")

# Change working directory to parent directory (where your data is stored)
try:
    if str.split(os.getcwd(),"\\")[2] == "brull":
        os.chdir('C:\\Users\\brull\\OneDrive - The Pennsylvania State University\\Team-8\\Data\\final_data_set')
    else:
        os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data\\final_data_set')
except:
    os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data\\final_data_set')

# Load cleaned file
file = 'final_data_set.csv'
df = pd.read_csv(file)
df = df.rename(columns = {'Unnamed: 0':'OriginalIndex'})
#type(df)
#len(df)
#df.head(5)

# # Load spacy large
# nlp = spacy.load("en_core_web_lg")

# # Sample cleaned Team 8 file
# #df_sample = df.sample(n=10000)
# #df_sample.columns

# # Apply spacy to the entire collection of reviews
# #docs = list(nlp.pipe(df_sample.reviewText))
# docs = list(nlp.pipe(df.reviewText))

# # function to extract word properties
# def extract_tokens_plus_meta(doc:spacy.tokens.doc.Doc):
#     """Extract tokens and metadata from individual spaCy doc."""
#     return [
#         (i.text, i.i, i.lemma_, i.ent_type_, i.tag_, 
#          i.dep_, i.pos_, i.is_stop, i.is_alpha, 
#          i.is_digit, i.is_punct) for i in doc
#     ]

# # function to apply extract_tokens_plus_meta to doc and output to dataframe
# def tidy_tokens(docs):
#     """Extract tokens and metadata from list of spaCy docs."""
    
#     cols = [
#         "doc_id", "token", "token_order", "lemma", 
#         "ent_type", "tag", "dep", "pos", "is_stop", 
#         "is_alpha", "is_digit", "is_punct"
#     ]
    
#     meta_df = []
#     for ix, doc in enumerate(docs):
#         meta = extract_tokens_plus_meta(doc)
#         meta = pd.DataFrame(meta)
#         try:
#             meta.columns = cols[1:]
#             meta = meta.assign(doc_id = ix).loc[:, cols]
#             meta_df.append(meta)
#         except:
#             print(str(ix)+" no data")
        
#     return pd.concat(meta_df)

# # Run new functions for 10k sample docs
# tidy_tokens(docs)

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# spacy.load('en_core_web_sm')


# #Check the size of the data

# print(df.index)

# #Set a seed and sample a random 10,000 rows of data

# #amanda_sample = df.sample(n=10000, random_state= 54321)

# #print(amanda_sample.index)

# # Expand the contractions

# # Dictionary of English Contractions
# contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
#                      "can't": "cannot","can't've": "cannot have",
#                      "'cause": "because","could've": "could have","couldn't": "could not",
#                      "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
#                      "don't": "do not","hadn't": "had not","hadn't've": "had not have",
#                      "hasn't": "has not","haven't": "have not","he'd": "he would",
#                      "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
#                      "how'd": "how did","how'd'y": "how do you","how'll": "how will",
#                      "I'd": "I would", "I'd've": "I would have","I'll": "I will",
#                      "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
#                      "it'd": "it would","it'd've": "it would have","it'll": "it will",
#                      "it'll've": "it will have", "let's": "let us","ma'am": "madam",
#                      "mayn't": "may not","might've": "might have","mightn't": "might not", 
#                      "mightn't've": "might not have","must've": "must have","mustn't": "must not",
#                      "mustn't've": "must not have", "needn't": "need not",
#                      "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
#                      "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
#                      "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
#                      "she'll": "she will", "she'll've": "she will have","should've": "should have",
#                      "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
#                      "that'd": "that would","that'd've": "that would have", "there'd": "there would",
#                      "there'd've": "there would have", "they'd": "they would",
#                      "they'd've": "they would have","they'll": "they will",
#                      "they'll've": "they will have", "they're": "they are","they've": "they have",
#                      "to've": "to have","wasn't": "was not","we'd": "we would",
#                      "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
#                      "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
#                      "what'll've": "what will have","what're": "what are", "what've": "what have",
#                      "when've": "when have","where'd": "where did", "where've": "where have",
#                      "who'll": "who will","who'll've": "who will have","who've": "who have",
#                      "why've": "why have","will've": "will have","won't": "will not",
#                      "won't've": "will not have", "would've": "would have","wouldn't": "would not",
#                      "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
#                      "y'all'd've": "you all would have","y'all're": "you all are",
#                      "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
#                      "you'll": "you will","you'll've": "you will have", "you're": "you are",
#                      "you've": "you have"}

# # Regular expression for finding contractions
# contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# # Function for expanding contractions
# def expand_contractions(text,contractions_dict=contractions_dict):
#   def replace(match):
#     return contractions_dict[match.group(0)]
#   return contractions_re.sub(replace, text)

# # Expanding Contractions in the reviews
# #amanda_sample['reviewText']=amanda_sample['reviewText'].apply(lambda x:expand_contractions(x))
# df['reviewText']=df['reviewText'].apply(lambda x:expand_contractions(x))

# print(df['reviewText'].head(50)) #take a look at a few after expansion

# # Lemmatization with stopwords removal

# # Loading model 
# nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

# #amanda_sample['reviewText']=amanda_sample['reviewText'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))
# df['reviewText']=df['reviewText'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))


# print(df['reviewText'].head(50)) #take a look at a few 

# #group our words by main product category

# grouped_sample = df[['main_cat','reviewText']].groupby(by='main_cat').agg(lambda x:' '.join(x))
# print(grouped_sample.head())

# # Create a Document Term Matrix - LEFT OFF HERE

df_random = df.sample(n=10000, random_state= 54321)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(analyzer='word')
data=cv.fit_transform(df_random['reviewText'].values.astype('U'))
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index=df_random.index
print(df_dtm.head(3))

print(df_dtm[['ship','shipping','package','damage','damaged','delay',
            'delayed','missing','ups','fedex','delivery','fast','slow',
            'crushed','burst','shredded','freight','supply','usps',
            'mail']].head(3))

print(df_dtm[['expensive','costly','credit','money','dollar','inexpensive',
        'bargain','rate','charge','fee',#'deduction',
        'discount',
        #'overcharge','surcharge',
        'tax','bill','invoice']].head(3))

print(df_dtm[['complaint','bad','excellent','manager','call','customer',
                    'service','chat','support','professional','unprofessional',
                    'rude','helpful','scam','friendly','email']].head(3))

print(df_dtm[['broken','missing','scratched','pristine',
                   'expected','bust','busted','torn',
                   'faulty','spoiled','flaw','flawed','flawless']].head(3))

# # Importing wordcloud for plotting word clouds and textwrap for wrapping longer text
# from wordcloud import WordCloud
# from textwrap import wrap
# import matplotlib.pyplot as plt


# # Function for generating word clouds
# def generate_wordcloud(data,title):
#   wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
#   plt.figure(figsize=(10,8))
#   plt.imshow(wc, interpolation='bilinear')
#   plt.axis("off")
#   plt.title('\n'.join(wrap(title,60)),fontsize=13)
#   plt.show()
  
# # Transposing document term matrix
# df_dtm=df_dtm.transpose()

# # Plotting word cloud for each product
# for index,product in enumerate(df_dtm.columns):
#   generate_wordcloud(df_dtm[product].sort_values(ascending=False),product)
  
 ############################################################################################ 
  
 # Check polorarity - This is not Working YET!!!

#from textblob import TextBlob
#grouped_sample['polarity']=grouped_sample['reviewText'].apply(lambda x:TextBlob(x).sentiment.polarity)
#def getSubjectivity(text):
#    return TextBlob(text).sentiment.subjectivity
  
 #Create a function to get the polarity
#def getPolarity(text):
#    return TextBlob(text).sentiment.polarity

#grouped_sample["subjectivity"] = grouped_sample['reviewText'].apply(getSubjectivity)
#grouped_sample["polarity"] = grouped_sample['reviewText'].apply(getPolarity)

#def getAnalysis(score):
#    if score < -.6:
#        return "Negative"
#    elif score >= -.6 and score < -.2:
#        return "Slightly Negative"
#    elif score >= -.2 and score < .2:
#        return "Neutral"
#    elif score >= .2 and score < .6:
#        return "Slightly Positive"
#    else:
#        return "Positive"

#grouped_sample["sentiment"] = grouped_sample["polarity"].apply(getAnalysis)