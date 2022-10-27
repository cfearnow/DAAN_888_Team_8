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
file = 'classified_file.csv'
df = pd.read_csv(file)
df = df.rename(columns = {'Unnamed: 0':'OriginalIndex'})
df['Classification'] = np.where(df['Class 0'].isin(['expected','clean','quality','material',
                                                    'advertise','rating','size','workmanship']),
                                'Product Quality', 
                                np.where(df['Class 0'].isin(['pleased','customer service','complain',
                                                             'contact','help','difficult','disclose',
                                                             'offensive']), 
                                         'Customer Service', 
                                         np.where(df['Class 0'].isin(['price','invoice','bargain'])
                                                  , 'Pricing & Finance'
                                                  , 'Shipping')
                                         )
                                )

file = 'BERT_sentimentLabelsandScores.csv'
bert = pd.read_csv(file)
bert = bert.rename(columns = {'Unnamed: 0':'OriginalIndex'})
bert_trim = bert[['BERT_FullSentiment', 'BERT_FullScore', 'idx']]


file = 'newbert.csv'
bert2 = pd.read_csv(file)
bert2 = bert2.rename(columns = {'Unnamed: 0':'OriginalIndex'})
bert2_trim = bert2[['BERT_FullSentiment','BERT_FullScore','idx']]


#df_random = df.sample(n=50000, random_state= 54321)
df_random = df
#df_random = df.sample(n=10)


CheckWords = ['ship','shipping','package','damage','damaged','delay',
             'delayed','missing','ups','fedex','delivery','fast','slow',
             'crushed','burst','shredded','freight','supply','usps','mail',
             'package','speed','issue', #Shipping
             
             'expensive','costly','credit','money','dollar','inexpensive',
             'bargain','rate','charge','fee','deduction','discount',
             'overcharge','surcharge','tax','bill','invoice', #Pricing & Finance
             
             'complaint','bad','excellent','manager','call','customer',
             'service','chat','support','professional','unprofessional',
             'rude','helpful','scam','friendly','email','pleased','complain',
             'contact','help','difficult','disclose','offensive', #Customer Service
             
             'broken','missing','scratched','pristine','expected','bust',
             'busted','torn','faulty','spoiled','flaw','flawed','flawless',
             'expected','clean','quality','material','advertise','rating',
             'size','workmanship'#Product Quality
             ]

test= pd.DataFrame(columns = ['reviewTextNew','idx'])
for i, row in df_random.iterrows():
    result = ''
    querywords = str(row['reviewText']).split()
    
    resultwords  = [str(word) for word in querywords if str(word).lower() in CheckWords]
    result = ' '.join(resultwords)
    
    test.loc[i] = [result,row['idx']]

#test2 = pd.merge(df_random,test, left_index = True, right_index = True)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(analyzer='word')

data=cv.fit_transform(test['reviewTextNew'].values.astype('U'))
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index=test.index
print(df_dtm.head(3))

# print(df_dtm[['ship','shipping','package','damage','damaged','delay',
#              'delayed','missing','ups','fedex','delivery','fast','slow',
#              'crushed','burst','shredded','freight','supply','usps','mail',
#              'package','speed','issue']].head(3))

# print(df_dtm[['expensive','costly','credit','money','dollar','inexpensive',
#              'bargain','rate','charge','fee','deduction','discount',
#              'overcharge','surcharge','tax','bill','invoice']].head(3))

# print(df_dtm[['complaint','bad','excellent','manager','call','customer',
#              'service','chat','support','professional','unprofessional',
#              'rude','helpful','scam','friendly','email','pleased','complain',
#              'contact','help','difficult','disclose','offensive']].head(3))

# print(df_dtm[['broken','missing','scratched','pristine','expected','bust',
#              'busted','torn','faulty','spoiled','flaw','flawed','flawless',
#              'expected','clean','quality','material','advertise','rating',
#              'size','workmanship']].head(3))


df_dtm['Shipping'] = (df_dtm['ship'].replace(np.nan, 0)+df_dtm['shipping'].replace(np.nan, 0)+df_dtm['package'].replace(np.nan, 0)+df_dtm['damage'].replace(np.nan, 0)+df_dtm['damaged'].replace(np.nan, 0)+df_dtm['delay'].replace(np.nan, 0)+
             df_dtm['delayed'].replace(np.nan, 0)+df_dtm['missing'].replace(np.nan, 0)+df_dtm['ups'].replace(np.nan, 0)+df_dtm['fedex'].replace(np.nan, 0)+df_dtm['delivery'].replace(np.nan, 0)+df_dtm['fast'].replace(np.nan, 0)+df_dtm['slow'].replace(np.nan, 0)+
             df_dtm['crushed'].replace(np.nan, 0)+df_dtm['burst'].replace(np.nan, 0)+df_dtm['shredded'].replace(np.nan, 0)+df_dtm['freight'].replace(np.nan, 0)+df_dtm['supply'].replace(np.nan, 0)+df_dtm['usps'].replace(np.nan, 0)+df_dtm['mail'].replace(np.nan, 0)+
             df_dtm['package'].replace(np.nan, 0)+df_dtm['speed'].replace(np.nan, 0)+df_dtm['issue'].replace(np.nan, 0))

df_dtm['PricingFinance'] = (df_dtm['expensive'].replace(np.nan, 0)+df_dtm['costly'].replace(np.nan, 0)+df_dtm['credit'].replace(np.nan, 0)+df_dtm['money'].replace(np.nan, 0)+df_dtm['dollar'].replace(np.nan, 0)+df_dtm['inexpensive'].replace(np.nan, 0)+
             df_dtm['bargain'].replace(np.nan, 0)+df_dtm['rate'].replace(np.nan, 0)+df_dtm['charge'].replace(np.nan, 0)+df_dtm['fee'].replace(np.nan, 0)+df_dtm['deduction'].replace(np.nan, 0)+df_dtm['discount'].replace(np.nan, 0)+
             df_dtm['overcharge'].replace(np.nan, 0)+df_dtm['surcharge'].replace(np.nan, 0)+df_dtm['tax'].replace(np.nan, 0)+df_dtm['bill'].replace(np.nan, 0)+df_dtm['invoice'].replace(np.nan, 0))

df_dtm['CustomerService'] = (df_dtm['complaint'].replace(np.nan, 0)+df_dtm['bad'].replace(np.nan, 0)+df_dtm['excellent'].replace(np.nan, 0)+df_dtm['manager'].replace(np.nan, 0)+df_dtm['call'].replace(np.nan, 0)+df_dtm['customer'].replace(np.nan, 0)+
             df_dtm['service'].replace(np.nan, 0)+df_dtm['chat'].replace(np.nan, 0)+df_dtm['support'].replace(np.nan, 0)+df_dtm['professional'].replace(np.nan, 0)+df_dtm['unprofessional'].replace(np.nan, 0)+
             df_dtm['rude'].replace(np.nan, 0)+df_dtm['helpful'].replace(np.nan, 0)+df_dtm['scam'].replace(np.nan, 0)+df_dtm['friendly'].replace(np.nan, 0)+df_dtm['email'].replace(np.nan, 0)+df_dtm['pleased'].replace(np.nan, 0)+df_dtm['complain'].replace(np.nan, 0)+
             df_dtm['contact'].replace(np.nan, 0)+df_dtm['help'].replace(np.nan, 0)+df_dtm['difficult'].replace(np.nan, 0)+df_dtm['disclose'].replace(np.nan, 0)+df_dtm['offensive'].replace(np.nan, 0))

df_dtm['ProductQuality'] = (df_dtm['broken'].replace(np.nan, 0)+df_dtm['missing'].replace(np.nan, 0)+df_dtm['scratched'].replace(np.nan, 0)+df_dtm['pristine'].replace(np.nan, 0)+df_dtm['expected'].replace(np.nan, 0)+df_dtm['bust'].replace(np.nan, 0)+
             df_dtm['busted'].replace(np.nan, 0)+df_dtm['torn'].replace(np.nan, 0)+df_dtm['faulty'].replace(np.nan, 0)+df_dtm['spoiled'].replace(np.nan, 0)+df_dtm['flaw'].replace(np.nan, 0)+df_dtm['flawed'].replace(np.nan, 0)+df_dtm['flawless'].replace(np.nan, 0)+
             df_dtm['expected'].replace(np.nan, 0)+df_dtm['clean'].replace(np.nan, 0)+df_dtm['quality'].replace(np.nan, 0)+df_dtm['material'].replace(np.nan, 0)+df_dtm['advertise'].replace(np.nan, 0)+df_dtm['rating'].replace(np.nan, 0)+
             df_dtm['size'].replace(np.nan, 0)+df_dtm['workmanship'].replace(np.nan, 0))

semifinal = pd.merge(test,df_dtm[['Shipping','PricingFinance','CustomerService','ProductQuality']], left_index = True, right_index = True)
semifinal2 = semifinal[['Shipping','PricingFinance','CustomerService','ProductQuality','idx']]
final = df.set_index('idx').join(semifinal2.set_index('idx'), on='idx', how='inner')
final['Validation'] = np.where(
    (final['Shipping'] > final['PricingFinance']) & (final['Shipping'] > final['CustomerService']) & (final['Shipping'] > final['ProductQuality']) & (final['Classification'] == 'Shipping'), 1, 
    np.where(
        (final['PricingFinance'] > final['Shipping']) & (final['PricingFinance'] > final['CustomerService']) & (final['PricingFinance'] > final['ProductQuality']) & (final['Classification'] == 'Pricing & Finance'), 1, 
        np.where(
            (final['CustomerService'] > final['PricingFinance']) & (final['CustomerService'] > final['Shipping']) & (final['CustomerService'] > final['ProductQuality']) & (final['Classification'] == 'Customer Service'), 1, 
            np.where(
                (final['ProductQuality'] > final['PricingFinance']) & (final['ProductQuality'] > final['CustomerService']) & (final['ProductQuality'] > final['Shipping']) & (final['Classification'] == 'Product Quality'), 1, 
                np.where(
                    (final['ProductQuality'] == 0) & (final['PricingFinance'] == 0) & (final['CustomerService'] == 0) & (final['Shipping'] == 0), np.nan, 0)))))

bert_final = pd.concat([bert_trim,bert2_trim])

combined = final.join(bert_final.set_index('idx'), on='idx',how='inner')
combined.reset_index(inplace=True)
combined = combined.rename(columns = {'index':'idx'})

combined.to_csv('combined_data.csv')

combined['Validation'].value_counts()
combined['Validation'].value_counts()[1]/(combined['Validation'].value_counts()[1]+combined['Validation'].value_counts()[0])