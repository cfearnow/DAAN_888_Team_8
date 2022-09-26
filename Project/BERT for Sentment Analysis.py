# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 19:19:03 2022

@author: cchee
"""

### This code was developed following the tutorial from this online article: 
## Sentiment Analysis of Tweets using BERT - Thinking Neuron
## https://thinkingneuron.com/sentiment-analysis-of-tweets-using-bert/

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

amanda_sample = clean_data.sample(n=200000, random_state= 12345)

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
FunctionBERTSentiment(inpText="I'm not sure this worked right")


# Calling BERT based sentiment score function for every review
amanda_sample['Sentiment']=amanda_sample['reviewText'].apply(FunctionBERTSentiment)
amanda_sample.head(200)

amanda_sample.to_csv('BERTsample_output.csv')

# Plot number of reviews by subcategory and sentiment
pivot = pd.pivot_table(amanda_sample, values = 'asin', index = 'sub_category', columns = 'Sentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'sub_category', rot = 50)
plt.xlabel("Sub Category")
plt.ylabel("Number of Reviews")
plt.title("Reviews by Sub Category and Sentiments")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

# Plot number of reviews by Star rating and sentiment
pivot = pd.pivot_table(amanda_sample, values = 'asin', index = 'overall', columns = 'Sentiment', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'overall', rot = 50)
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.title("Reviews by Overall Star Rating and Sentiments")
plt.legend(title="Sentiment", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

############### what needs to be done to figure out accuracy ###########################

### https://discuss.pytorch.org/t/f1-score-in-pytorch-for-evaluation-of-the-bert/144382

def evaluate(model, val_dataloader):
    """
    After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    f1_weighted = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

        # Calculate the f1 weighted score
        f1_metric = F1Score('weighted') 
        f1_weighted = f1_metric(preds, b_labels)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    f1_weighted = np.mean(f1_weighted)

    return val_loss, val_accuracy, f1_weighted 





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
