# Import necessary packages
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import datetime
import json
import gzip
import os
import csv
import io
import re
import string
import glob
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from ast import literal_eval
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#%%
# Set directories

# Change working directory to parent directory (where your data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

# Directory to chunked results
chunked_path = './Data/chunked'

#%%
# Create global variables for use in analysis
files = glob.glob('Data\Sent_Analysis\Clean_Data\*.parquet.gz')

#%%
# Load data
clean = pd.read_parquet(files, engine='pyarrow')

# Take a closer look at the data
clean_head = clean.head(50)

# Create a new index column to ensure no duplicates
clean['idx'] = range(1, len(clean) + 1)

# Save again as parquet
clean.to_parquet('presentiment_whole.parquet.gz', compression='gzip')

# Extract only review columns to assist with file size
clean_reviews = clean[['idx', 'reviewText', 'summary', 'original_text', 'original_summary' ]].copy()

# Save review column file as parquet
clean_reviews.to_parquet('presentiment_reviews.parquet.gz', compression='gzip')

#%%
def chunk_data(file_name):
    
    n = 100
    chunk_reviews = np.array_split(file_name, n)
    
    # Check if augmented directory exists
    folder_check = os.path.isdir(chunked_path)

    if not folder_check:
        os.makedirs(chunked_path)
        print("created folder : ", chunked_path)
    else:
        print(chunked_path, "already exists.")
        
    for i in range(len(chunk_reviews)):
        varname = 'chunk_' + str(i) 
        new_name = varname + '.csv'
        folder_path = os.path.join(chunked_path, new_name)
        exists_check = os.path.isdir(folder_path)
        
        if not exists_check:
            varname = chunk_reviews[i]
            #varname['tokenized_sents'] = varname.apply(lambda row: nltk.word_tokenize(row['reviewText']), axis=1)
            varname.to_csv(folder_path, encoding='utf-8')
            print("created file : ", new_name)
        else:
            print(new_name, " already exists.")

chunk_data(clean_reviews)

#%%

file_chunks = glob.glob(chunked_path + '/*.csv')

#####    UPDATE BEFORE RUNNING    ###########
results_folder = 'Data/Sent_Analysis/Results/orig_summary/'
results_name = 'orig_summary_sent_'
col_names = ['idx', 'original_summary']
text_field = 'original_summary'

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

def sentiment(file_path, sentiment_column):
            
    folder = results_folder
    
    folder_check = os.path.isdir(folder)
    
    if not folder_check:
        os.makedirs(folder)
        print("created folder : ", folder)
    else:
        print(folder, "already exists.")
        
    for file in file_path:
        emp_str = ""
        for m in file:
            if m.isdigit():
                emp_str = emp_str + m
        name = results_name + emp_str
        
        file_name = (results_folder + name +'.csv')
        file_check = os.path.isdir(file_name)
        
        if not file_check:
            print('Reading: ' + file)
            x = pd.read_csv(file, usecols = col_names)
            print('Analyzing: ' + file)
            y = get_sentiment_scores(x, sentiment_column)
            print('Writing: ' + name)
            y.to_csv(file_name, encoding='utf-8')
            print("created file : ", name)
        else:
            print(name, " already exists.")
            
sentiment(file_chunks, text_field)

###############################################################################
