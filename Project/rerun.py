# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 07:47:54 2022

@author: chris
"""
# Import necessary packages
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn import metrics

#%%
# Set directories

# Change working directory to parent directory (where your data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

# Directory to chunked results
final = pd.read_csv('final_data_set.csv')

#%%

sid_analyzer = SentimentIntensityAnalyzer()
#############################################

def get_sentiment(text:str, analyser,desired_type:str='pos'):
    # Get sentiment from text
    sentiment_score = analyser.polarity_scores(text)
    return sentiment_score[desired_type]

# Get Sentiment scores
def get_sentiment_scores(df,data_column):
    df[f'{data_column} Positive Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'pos'))
    df[f'{data_column} Negative Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neg'))
    df[f'{data_column} Neutral Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neu'))
    df[f'{data_column} Compound Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'compound'))
    return df

get_sentiment_scores(final, 'original_text')

x = final.head()



###############################################################################

orig_conditions = [
    (final['original_text Compound Sentiment Score'] > 0.5),
    (final['original_text Compound Sentiment Score'] < -0.5),
    (final['original_text Compound Sentiment Score'] <= 0.5) & (final['original_text Compound Sentiment Score'] >= -0.5)
]

values = ['Positive', 'Negitive', 'Neutral']

final['orig_sentiment'] = np.select(orig_conditions, values)  

x = final.head(50)

# Plot number of reviews by subcategory and rating
text_pivot = pd.pivot_table(final, values = 'title', index = 'overall', columns = 'orig_sentiment', aggfunc= 'count')
text_pivot = text_pivot.reset_index()
text_pivot.plot.bar(x = 'overall', y = ['Negitive', 'Neutral', 'Positive'], rot = 50)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Text Reviews by Overall Star Rating and Sentiment")
plt.legend(title="Rating", loc='upper left', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

metrics.precision_recall_fscore_support(final['sentiment'], final['orig_sentiment'], average='weighted')

cm = metrics.confusion_matrix(final['sentiment'], final['orig_sentiment']) 

final['orig_sentiment'].value_counts()

final['sentiment'].value_counts()

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

print(accuracy_score(final['sentiment'], final['orig_sentiment'], average='weighted'))
print(recall_score(final['sentiment'], final['orig_sentiment'], average='weighted'))
print(precision_score(final['sentiment'], final['orig_sentiment'], average='weighted'))
print(f1_score(final['sentiment'], final['orig_sentiment'], average='weighted'))
print(classification_report(final['sentiment'], final['orig_sentiment']))
