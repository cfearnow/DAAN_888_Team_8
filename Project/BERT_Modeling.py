# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 09:30:01 2022

@author: brull
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import csv
from sklearn import metrics
from transformers import pipeline


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

try:
    if str.split(os.getcwd(),"\\")[2] == "brull":
        os.chdir('C:\\Users\\brull\\OneDrive - The Pennsylvania State University\\Team-8\\Data\\Output')
    else:
        os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data')
except:
    os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data')

df = pd.read_parquet("Toys_and_Games.parquet", engine='pyarrow')

df2 = df[df['reviewText'].str.split().str.len().lt(513)]

SentimentClassifier = pipeline("sentiment-analysis")

df_sample = df.sample(n=1000)

def FunctionBERTSentimentTxt(inpText):
  return(SentimentClassifier(inpText)[0]['label'])

def FunctionBERTSentimentScore(inpText):
    return(SentimentClassifier(inpText)[0]['score'])
 
df_sample['ReviewSentimentTxt']=df_sample['reviewText'].apply(FunctionBERTSentimentTxt)
df_sample['ReviewSentimentScore']=df_sample['reviewText'].apply(FunctionBERTSentimentScore)
df_sample['SummarySentimentTxt']=df_sample['summary'].apply(FunctionBERTSentimentTxt)
df_sample['SummarySentimentScore']=df_sample['summary'].apply(FunctionBERTSentimentScore)
df_sample.head(10)
