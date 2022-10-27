# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:22:28 2022

@author: brull
"""
###########################################################################
#To run this application, open Conda Terminal and run the following command:
#streamlit run "C:\Users\brull\OneDrive - The Pennsylvania State University\Team-8\Code Repo\DAAN_888_Team_8\Project\ModelStreamlit.py"
###########################################################################        

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
from pynvml import *
from tqdm import tqdm
import seaborn as sns

# Packages for NLP
import spacy
from spacy.tokens import Span
from spacy import displacy
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import datetime
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from transformers import pipeline
import torch
from streamlit.components.v1 import html

torch.cuda.is_available()

##############################################################################
def txt_clean(file_name, file_column):
    # If null change to "" to prevent errors
    file_name[file_column] = file_name[file_column].fillna(value='')
    # Make all strings lowercase
    file_name[file_column] = file_name[file_column].apply(lambda x: x.lower())
    # Remove any numerical digits
    file_name[file_column] = file_name[file_column].apply(lambda x: re.sub('\w*\d\w*','', x))
    # Remove all punctuation
    file_name[file_column] = file_name[file_column].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
    # Remove any extra spaces
    file_name[file_column] = file_name[file_column].apply(lambda x: re.sub(' +',' ',x))
    return(file_name)

def str_clean(file_name, file_column):
    # Remove line breaks
    file_name[file_column] = file_name[file_column].str.replace('(<br/>)', '')
    file_name[file_column] = file_name[file_column].str.replace('(<br>)', '')
    file_name[file_column] = file_name[file_column].str.replace('(</br>)', '')
    # Remove noise text
    file_name[file_column] = file_name[file_column].str.replace('(<a).*(>).*(</a>)', '')
    # Remove ampersand
    file_name[file_column] = file_name[file_column].str.replace('(&amp)', '')
    # Remove greather than
    file_name[file_column] = file_name[file_column].str.replace('(&gt)', '')
    # Remove less than
    file_name[file_column] = file_name[file_column].str.replace('(&lt)', '')
    # Remove unicode hard space or a no-break space
    file_name[file_column] = file_name[file_column].str.replace('(\xa0)', ' ')
    return file_name

def get_data():
    return []

SentimentClassifier = pipeline("sentiment-analysis")

def FunctionBERTSentiment(inpText):
  return(SentimentClassifier(inpText)[0]['label'])

def FunctionBERTSentimentScore(inpText):
  return(SentimentClassifier(inpText)[0]['score'])
##############################################################################

streamlitlist = []#pd.DataFrame(columns=['reviewText'])

#Add application title
st.markdown("<h1 style='text-align: center;'>Ad-Hoc Review Sentiment & Classification</h1>", unsafe_allow_html=True)

#Create a form (this will allow all variables created to be auto-passed when using form-submit button)
with st.form(key="my_form"):
    #Create an input text section
    doc = st.text_area(
               "Paste your text below (max 500 words)",
               height=510,
           )
    
    MAX_WORDS = 500
    res = len(re.findall(r"\w+", doc))
    if res > MAX_WORDS:
        st.warning(
            "⚠️ Your text contains "
            + str(res)
            + " words."
            + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
        )
    
    doc = doc[:MAX_WORDS]
    
    uploaded_file = st.file_uploader("Optional: Choose a CSV file", type = 'csv')
    
    
    if st.form_submit_button(label="✨ Check the data!"):
        if uploaded_file is not None:
            # Can be used wherever a "file-like" object is accepted:
            filedf = pd.read_csv(uploaded_file)
            filedf.columns = filedf.columns.str.lower()
            if 'reviewtext' not in filedf.columns:
                st.error('🚨Your uploaded file must include a \"reviewText\" column. Ignoring file input🚨')
                filedf = None
            else:
                filedf = pd.DataFrame(filedf.rename(columns={'reviewtext':'reviewText'})['reviewText'])
        else:
            filedf = None
        if doc == '' and uploaded_file is None:
            st.error('🚨Cannot review with no text submitted, exiting app🚨')
            st.stop()
        elif doc == '' and 'reviewText' not in filedf.columns:
            st.error('🚨Cannot review with no text submitted, exiting app🚨')
            st.stop()
        streamlitlist.append({"reviewText": doc})
        inputdf = pd.DataFrame(streamlitlist, columns = ['reviewText'])
        if filedf is not None and doc != '':
            streamlitdf = pd.concat([inputdf,filedf], axis = 0)
        elif doc == '':
            streamlitdf = filedf
        else:
            streamlitdf = inputdf
        streamlitdf = streamlitdf.rename(columns={'reviewText':'originalEntry'})
        st.dataframe(streamlitdf)
        all_clean = str_clean(streamlitdf, 'originalEntry')
        final_clean = txt_clean(all_clean, 'originalEntry')
        final_clean['BERT_Sentiment'] = final_clean['originalEntry'].apply(FunctionBERTSentiment)
        final_clean['BERT_Score'] = final_clean['originalEntry'].apply(FunctionBERTSentimentScore)
        final_clean = final_clean.rename(columns={'originalEntry':'cleanedEntry'})
        st.dataframe(final_clean)
        test_file = final_clean
        test_file2 = test_file[test_file['cleanedEntry'].notnull()]
        classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", 
                      device = 0)

        candidate_labels = ['package',
                            'price',
                            'expected',
                            'pleased',
                            'clean',
                            'quality',
                            'invoice',
                            'customer service',
                            'damage',
                            'speed',
                            'bargain',
                            'complain',
                            'contact',
                            'shipping',
                            'help',
                            'difficult',
                            'disclose',
                            'offensive',
                            'material',
                            'advertise',
                            'rating',
                            'size',
                            'workmanship',
                            'issue'
                            ]
        predictedCategories = []
        finalfinallist = {'Class 0':[],'Rating 0':[],'Class 1':[],'Rating 1':[],'Class 2':[],'Rating 2':[],'Class 3':[],'Rating 3':[],'Class 4':[],'Rating 4':[]}
        for i in test_file2.index:
            text = test_file2.iloc[i,]['cleanedEntry']
            #idx_num = x.iloc[i,]['idx']
            res = classifier(text, candidate_labels, multi_label=True)#setting multi-class as True
            labels = res['labels'] 
            scores = res['scores'] #extracting the scores associated with the labels
            res_dict = {label : score for label,score in zip(labels, scores)}
            sorted_dict = dict(sorted(res_dict.items(), key=lambda test_file2:test_file2[1],reverse = True)) #sorting the dictionary of labels in descending order based on their score
            categories  = []
            for x, (k,v) in enumerate(sorted_dict.items()):
                if(x > 4): #storing only the best 5 predictions
                    break
                else:
                    categories.append([k, v])
            predictedCategories.append(categories)
            final_list = {'Class 0':[predictedCategories[i][0][0]]
                          ,'Rating 0':[predictedCategories[i][0][1]]
                          ,'Class 1':[predictedCategories[i][1][0]]
                          ,'Rating 1':[predictedCategories[i][1][1]]
                          ,'Class 2':[predictedCategories[i][2][0]]
                          ,'Rating 2':[predictedCategories[i][2][1]]
                          ,'Class 3':[predictedCategories[i][3][0]]
                          ,'Rating 3':[predictedCategories[i][3][1]]
                          ,'Class 4':[predictedCategories[i][4][0]]
                          ,'Rating 4':[predictedCategories[i][4][1]]
                          }
            finalfinallist['Class 0'].append(final_list['Class 0'][0])
            finalfinallist['Rating 0'].append(final_list['Rating 0'][0])
            finalfinallist['Class 1'].append(final_list['Class 1'][0])
            finalfinallist['Rating 1'].append(final_list['Rating 1'][0])
            finalfinallist['Class 2'].append(final_list['Class 2'][0])
            finalfinallist['Rating 2'].append(final_list['Rating 2'][0])
            finalfinallist['Class 3'].append(final_list['Class 3'][0])
            finalfinallist['Rating 3'].append(final_list['Rating 3'][0])
            finalfinallist['Class 4'].append(final_list['Class 4'][0])
            finalfinallist['Rating 4'].append(final_list['Rating 4'][0])
        final_df = pd.DataFrame.from_dict(finalfinallist)
        st.dataframe(final_df)
        df = final_df[['Class 0','Rating 0']]
        #df['Class 0'] = df['Class 0'].astype('string')
        df['Classification'] = np.where(df['Class 0'].isin(['expected','clean','quality','material','advertise','rating','size','workmanship']),'Product Quality', np.where(df['Class 0'].isin(['pleased','customer service','complain','contact','help','difficult','disclose','offensive']), 'Customer Service',np.where(df['Class 0'].isin(['price','invoice','bargain']), 'Pricing & Finance', 'Shipping')))
        st.dataframe(df)
        classification = pd.DataFrame(df['Classification'])
        final = pd.merge(pd.merge(pd.merge(streamlitdf, final_clean,left_index=True, right_index=True),final_df,left_index=True, right_index=True),classification,left_index=True, right_index=True)
        st.dataframe(final)
        csv_file = final.to_csv().encode('utf-8')
     
#Using a try except block to keep the download button hidden until the form has been ran
try:
    if final is not None:
        st.download_button(
        label="Download data as CSV",
        data=csv_file,
        file_name='SentimentClassificationOutput_'+str(datetime.now().strftime("%Y%m%d_%H%M%S"))+'.csv',
        mime='text/csv',
        )
except:
    pass
        

