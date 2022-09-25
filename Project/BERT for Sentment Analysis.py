# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 19:19:03 2022

@author: cchee
"""

import pandas as pd
import json
import gzip
import os
import re
import string


from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

import tensorflow as tf

# importing the pipeline module
from transformers import pipeline


# This is Amanda's attempt at BERT Sentiment Analysis

# See current working directory
os.getcwd()


# Change working directory to parent directory (where your data is stored)

os.chdir('C:\\users\\cchee\\OneDrive\\Desktop')

#read the file

clean_data = pd.read_parquet('team8_initial_clean.parquet.gz', engine ='pyarrow')

#Check the size of the data

print(clean_data.index)

#Set a seed and sample a random 250,000 rows of data

amanda_sample = clean_data.sample(n=1000, random_state= 12345)

print(amanda_sample.index)

# Build the pretrained BERT model with sequence classifier and tokenizer

# Downloading the sentiment analysis model
SentimentClassifier = pipeline("sentiment-analysis")


# Calling the sentiment analysis function for 3 sentences - test this.
SentimentClassifier(["I hope we get all these concepts! Its killing the neurons of our brain",
                     "We had a nice experience in this trip",
                     "Houston we have a problem"
                      ])

# Try the analysis on the Amazon review data

# Defining a function to call for the whole dataframe
def FunctionBERTSentiment(inpText):
  return(SentimentClassifier(inpText)[0]['label'])
 
# Calling the function - just test it
FunctionBERTSentiment(inpText="Houston we have a problem")


# Calling BERT based sentiment score function for every review
amanda_sample['Sentiment']=amanda_sample['reviewText'].apply(FunctionBERTSentiment)
amanda_sample.head(200)




# Alex's funtion - can we use something like this???

def getAnalysis(score):
    if score < -.6:
        return "Negative"
    elif score >= -.6 and score < -.2:
        return "Slightly Negative"
    elif score >= -.2 and score < .2:
        return "Neutral"
    elif score >= .2 and score < .6:
        return "Slightly Positive"
    else:
        return "Positive"
