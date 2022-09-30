# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 19:19:03 2022

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


# BERT Sentiment analysis

# See current working directory

os.getcwd()


# Change working directory to parent directory (where your data is stored)

os.chdir('C:\\users\\cchee\\OneDrive\\Desktop')

#read the file

clean_data = pd.read_parquet('new_presentiment_whole.parquet.gz', engine ='pyarrow')

print(clean_data.index) #Check the size of the data

# make sure the raw review text has less than 512 tokens for BERT to run

clean_data['smallReview'] = clean_data['original_text'].apply(lambda x: len(x.split()))


#Set a seed and sample a random 200,000 rows of data

BERT_sample = clean_data.sample(n=200000, random_state= 321)

print(BERT_sample.index) # double check the size of the sample

BERT_sample.to_csv('BERTsample_data.csv') # store to working directory


# Make sure the summary and full text reviews are strings

BERT_sample['original_summary'] = BERT_sample['original_summary'].astype(str)
BERT_sample['original_text'] = BERT_sample['original_text'].astype(str)

# Build the pretrained BERT model with sequence classifier and tokenizer

SentimentClassifier = pipeline("sentiment-analysis") # Downloading the sentiment analysis model


# Calling the sentiment analysis function on some test sentences

SentimentClassifier(["We had a nice experience in this trip",
                     "Houston we have a problem", 
                     "I hate this store"
                      ])

# Try the analysis on the Amazon review data

# Defining a function to call the label for the whole dataframe

def FunctionBERTSentiment(inpText):
  return(SentimentClassifier(inpText)[0]['label'])
 
# Calling the label function - just test it

FunctionBERTSentiment(inpText="This is fun!")

# Defining a function to call the score for the whole dataframe

def FunctionBERTSentimentScore(inpText):
  return(SentimentClassifier(inpText)[0]['score'])

# Calling the score function - just test it

FunctionBERTSentimentScore(inpText="This is fun!")


# Calling BERT based sentiment label function for every full review

BERT_sample['FullSentiment']=BERT_sample['original_text'].apply(FunctionBERTSentiment)
BERT_sample.head(200)

BERT_sample.to_csv('BERTfull_sample_data.csv') # store to working directory

# Calling BERT based sentiment label function for every summary review

BERT_sample['SummSentiment']=BERT_sample['original_summary'].apply(FunctionBERTSentiment)
BERT_sample.head(200)

BERT_sample.to_csv('BERTsentiments_all.csv') # Write to working directory

# Calling BERT based sentiment score function for every full review

BERT_sample['FullScore']=BERT_sample['original_text'].apply(FunctionBERTSentimentScore)
BERT_sample.head(200)

BERT_sample.to_csv('BERTsentiments_fullScore.csv') # Write to working directory

# Calling BERT based sentiment score function for every full review

BERT_sample['SummScore']=BERT_sample['original_summary'].apply(FunctionBERTSentimentScore)
BERT_sample.head(200)

BERT_sample.to_csv('BERTsentiment_all.csv')  # write to working directory

# Create bar charts of reviews by sentiment, product category, and number of stars

# Plot number of reviews by subcategory and sentiment on full review

pivot = pd.pivot_table(BERT_sample, values = 'asin', index = 'sub_category', columns = 'FullSentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'sub_category', rot = 50)
plt.xlabel("Sub Category")
plt.ylabel("Number of Reviews")
plt.title("Full Text Reviews by Sub Category and Sentiment")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

# Plot number of full reviews by Star rating and sentiment

pivot = pd.pivot_table(BERT_sample, values = 'asin', index = 'overall', columns = 'FullSentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'overall', rot = 50)
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.title("Full Text Reviews by Overall Star Rating and Sentiment")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

# Plot number of summary reviews by subcategory and sentiment on 

pivot = pd.pivot_table(BERT_sample, values = 'asin', index = 'sub_category', columns = 'SummSentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'sub_category', rot = 50)
plt.xlabel("Sub Category")
plt.ylabel("Number of Reviews")
plt.title("Summary Reviews by Sub Category and Sentiment")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

# Plot number of summary reviews by Star rating and sentiment

pivot = pd.pivot_table(BERT_sample, values = 'asin', index = 'overall', columns = 'SummSentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'overall', rot = 50)
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.title("Summary Reviews by Overall Star Rating and Sentiment")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

# Create the box plots of scores for full text reviews

plt.boxplot(BERT_sample['FullScore'])
plt.ylabel("Sentiment Score")
plt.tick_params(right = False, labelbottom= False, bottom = False)
plt.title("Sentiment Score for Full Text Reviews") # Change title to match dataset
plt.show()


# Create the box plots of scores for summary text reviews

plt.boxplot(BERT_sample['SummScore'])
plt.ylabel("Sentiment Score")
plt.tick_params(right = False, labelbottom= False, bottom = False)
plt.title("Sentiment Score for Summary Reviews") # Change title to match dataset
plt.show()

