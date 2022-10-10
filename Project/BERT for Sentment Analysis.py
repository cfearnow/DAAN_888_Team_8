# -*- coding: utf-8 -*-
"""
Created on Sat Oct 8 19:19:03 2022

@author: cchee
"""

### This code was developed following the tutorial from this online article: 
## Sentiment Analysis of Tweets using BERT - Thinking Neuron
## https://thinkingneuron.com/sentiment-analysis-of-tweets-using-bert/

# Import Packages

import pandas as pd
import json
import gzip
import os
import re
import string
import numpy as np
import matplotlib.pyplot as plt

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from transformers import pipeline

# Check to make sure GPU processing can take place

print("Number of GPUs available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# This is the BERT sentiment analysis on the larger data sample containing 400k rows

os.getcwd() # See current working directory

os.chdir('C:\\users\\cchee\\OneDrive\\Desktop') # Change working directory to parent directory (where your data is stored)


data_sample = pd.read_parquet('new_presentiment_whole.parquet.gz', engine ='pyarrow') # read the file

print(data_sample.index) #Check the size of the data

# Make sure the summary and full text reviews are strings

data_sample['original_summary'] = data_sample['original_summary'].astype(str)
data_sample['original_text'] = data_sample['original_text'].astype(str)

# Build the pretrained BERT model with sequence classifier and tokenizer

SentimentClassifier = pipeline("sentiment-analysis") # Downloading the sentiment analysis model


# Calling the sentiment analysis classifier on some test sentences

SentimentClassifier(["We had a nice experience on this trip",
                     "Houston we have a problem", 
                     "I hate this store"
                      ])

# Complete the BERT sentiment analysis on the Amazon review data

# Define a function to call the sentiment label for the whole dataframe

def FunctionBERTSentiment(inpText):
  return(SentimentClassifier(inpText)[0]['label'])
 
# Calling the label function - just test it

FunctionBERTSentiment(inpText="This is fun!")

# Defining a function to call the sentiment score for the whole dataframe

def FunctionBERTSentimentScore(inpText):
  return(SentimentClassifier(inpText)[0]['score'])

# Calling the score function - just test it

FunctionBERTSentimentScore(inpText="This is fun!")


# Calling BERT based sentiment label function for every full review

data_sample['BERT_FullSentiment']=data_sample['original_text'].apply(FunctionBERTSentiment)
data_sample.head(200)

data_sample.to_csv('BERTfull_sample_data.csv') # store to working directory

# Calling BERT based sentiment label function for every summary review

data_sample['BERT_SummSentiment']=data_sample['original_summary'].apply(FunctionBERTSentiment)
data_sample.head(200)

data_sample.to_csv('BERTsentiments_allFull.csv') # Write to working directory

# Calling BERT based sentiment score function for every full review

data_sample['BERT_FullScore']=data_sample['original_text'].apply(FunctionBERTSentimentScore)
data_sample.head(200)

data_sample.to_csv('BERTsentiments_fullScore.csv') # Write to working directory

# Calling BERT based sentiment score function for every summary review

data_sample['BERT_SummScore']=data_sample['original_summary'].apply(FunctionBERTSentimentScore)
data_sample.head(200)

data_sample.to_csv('BERTsentiment_all.csv')  # write to working directory

# Create bar charts of reviews by sentiment, product category, and number of stars

# Plot number of reviews by subcategory and sentiment on full review

pivot = pd.pivot_table(data_sample, values = 'asin', index = 'sub_category', columns = 'BERT_FullSentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'sub_category', rot = 50)
plt.xlabel("Sub Category")
plt.ylabel("Number of Reviews")
plt.title("Full Text Reviews by Sub Category and Sentiment")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

# Plot number of full reviews by Star rating and sentiment

pivot = pd.pivot_table(data_sample, values = 'asin', index = 'overall', columns = 'BERT_FullSentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'overall', rot = 50)
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.title("Full Text Reviews by Overall Star Rating and Sentiment")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

# Plot number of summary reviews by subcategory and sentiment

pivot = pd.pivot_table(data_sample, values = 'asin', index = 'sub_category', columns = 'BERT_SummSentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'sub_category', rot = 50)
plt.xlabel("Sub Category")
plt.ylabel("Number of Reviews")
plt.title("Summary Reviews by Sub Category and Sentiment")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

# Plot number of summary reviews by Star rating and sentiment

pivot = pd.pivot_table(data_sample, values = 'asin', index = 'overall', columns = 'BERT_SummSentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'overall', rot = 50)
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.title("Summary Reviews by Overall Star Rating and Sentiment")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()
