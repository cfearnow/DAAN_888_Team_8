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
import xlsxwriter

# Packages for NLP
import spacy
from spacy.tokens import Span
from spacy import displacy
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from transformers import pipeline
import torch
from streamlit.components.v1 import html

try:
    if str.split(os.getcwd(),"\\")[2] == "brull":
        os.chdir('C:\\Users\\brull\\OneDrive - The Pennsylvania State University\\Team-8\\Live Demo')
    else:
        os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\')
except:
    os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\')

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
# from PIL import Image
# image = Image.open('Amazon Logo.png')

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.write("")

# with col2:
#     st.image(image, width = 150)

# with col3:
#     st.write("")

st.markdown("<h2 style='text-align: center;'><img src='https://d3.harvard.edu/platform-digit/wp-content/uploads/sites/2/2018/03/rerre.png' width = '250' height = '100'></h2>", unsafe_allow_html=True)


#Create a form (this will allow all variables created to be auto-passed when using form-submit button)
with st.form(key="my_form"):
    #Create an input text section
    brand = st.text_input("Associated Brand")
    title = st.text_input("Item name")
    category = st.selectbox('Item Category',('Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies','Toys_and_Games','Other'))
    reviewtime = str(datetime.now().strftime("%m/%d/%Y"))
    overall = st.slider('Overall Star Rating',1,5,3)
    asin = st.text_input('Amazon Unique Identifier')
    
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
    
    #title,brand,main_cat,price,asin,verified,reviewTime,reviewText,summary,overall,sub_category,price_adj,original_text,original_summary,idx
    
    if st.form_submit_button(label="✨ Check the data!"):
        if uploaded_file is not None:
            # Can be used wherever a "file-like" object is accepted:
            filedf = pd.read_csv(uploaded_file)
            filedf.columns = filedf.columns.str.lower()
            if 'reviewtext' not in filedf.columns:
                st.error('🚨Your uploaded file must include a \"reviewText\" column. Ignoring file input🚨')
                filedf = None
            else:
                if 'title' in filedf.columns:
                    if filedf['title'][0] == 'OPTIONAL: This is the name of the item being reviewed':
                        filedf = filedf.drop([0]).reset_index()
                filedf = filedf.rename(columns={'reviewtext':'reviewText'})
                if 'brand' not in filedf.columns:
                    if brand != '':
                        filedf['brand'] = brand
                    else:
                        filedf['brand'] = ''
                if 'title' not in filedf.columns:
                    if title != '':
                        filedf['title'] = title
                    else:
                        filedf['title'] = ''
                if 'sub_category' not in filedf.columns:
                    if category != '':
                        filedf['sub_category'] = category
                    else:
                        filedf['sub_category'] = ''
                if 'reviewtime' not in filedf.columns:
                    filedf['reviewTime'] = reviewtime
                else:
                    filedf = filedf.rename(columns={'reviewtime':'reviewTime'})
                if 'overall' not in filedf.columns:
                    if overall != '':
                        filedf['overall'] = overall
                    else:
                        filedf['overall'] = 0
                if 'asin' not in filedf.columns:
                    if asin != '':
                        filedf['asin'] = asin
                    else:
                        filedf['asin'] = ''
                    filedf = filedf[['title','brand','asin','reviewTime','reviewText','overall','sub_category']]
                else:
                    filedf = filedf[['title','brand','asin','reviewTime','reviewText','overall','sub_category']]
        else:
            filedf = None
        if doc == '' and uploaded_file is None:
            st.error('🚨Cannot review with no text submitted, exiting app🚨')
            st.stop()
        elif doc == '' and 'reviewText' not in filedf.columns:
            st.error('🚨Cannot review with no text submitted, exiting app🚨')
            st.stop()
        streamlitlist.append({"title":title,"brand":brand,"asin":asin,"reviewTime":reviewtime,"reviewText": doc,"overall":overall,"sub_category":category})
        inputdf = pd.DataFrame(streamlitlist, columns = ['title','brand','asin','reviewTime','reviewText','overall','sub_category'])
        if filedf is not None and doc != '':
            streamlitdf = pd.concat([inputdf,filedf], axis = 0,ignore_index=True)
        elif doc == '':
            streamlitdf = filedf
        else:
            streamlitdf = inputdf
        streamlitdf = streamlitdf.rename(columns={'reviewText':'originalEntry'})
        streamlitdf['overall'] = streamlitdf['overall'].astype('int')
        streamlitdf['overall'] = np.where(streamlitdf['overall'] > 5, '',np.where(streamlitdf['overall'] < 1, '', streamlitdf['overall']))
        st.dataframe(streamlitdf)
        all_clean = str_clean(streamlitdf, 'originalEntry')
        final_clean = txt_clean(all_clean, 'originalEntry')
        final_clean['BERT_FullSentiment'] = final_clean['originalEntry'].apply(FunctionBERTSentiment)
        final_clean['BERT_FullScore'] = final_clean['originalEntry'].apply(FunctionBERTSentimentScore)
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
        df = final_df#[['Class 0','Rating 0']]
        df['Classification1'] = np.where(df['Rating 0'] >= .75, np.where(df['Class 0'].isin(['expected','clean','quality','material','advertise','rating','size','workmanship']),'DAAN888ProductQuality@gmail.com', np.where(df['Class 0'].isin(['pleased','customer service','complain','contact','help','difficult','disclose','offensive']), 'DAAN888CustomerService@gmail.com',np.where(df['Class 0'].isin(['price','invoice','bargain']), 'DAAN888PricingFinance@gmail.com', 'DAAN888Shipping@gmail.com'))),'')
        df['Classification2'] = np.where(df['Rating 1'] >= .75, np.where(df['Class 1'].isin(['expected','clean','quality','material','advertise','rating','size','workmanship']),'DAAN888ProductQuality@gmail.com', np.where(df['Class 1'].isin(['pleased','customer service','complain','contact','help','difficult','disclose','offensive']), 'DAAN888CustomerService@gmail.com',np.where(df['Class 1'].isin(['price','invoice','bargain']), 'DAAN888PricingFinance@gmail.com', 'DAAN888Shipping@gmail.com'))),'')
        df['Classification3'] = np.where(df['Rating 2'] >= .75, np.where(df['Class 2'].isin(['expected','clean','quality','material','advertise','rating','size','workmanship']),'DAAN888ProductQuality@gmail.com', np.where(df['Class 2'].isin(['pleased','customer service','complain','contact','help','difficult','disclose','offensive']), 'DAAN888CustomerService@gmail.com',np.where(df['Class 2'].isin(['price','invoice','bargain']), 'DAAN888PricingFinance@gmail.com', 'DAAN888Shipping@gmail.com'))),'')
        df['Classification4'] = np.where(df['Rating 3'] >= .75, np.where(df['Class 3'].isin(['expected','clean','quality','material','advertise','rating','size','workmanship']),'DAAN888ProductQuality@gmail.com', np.where(df['Class 3'].isin(['pleased','customer service','complain','contact','help','difficult','disclose','offensive']), 'DAAN888CustomerService@gmail.com',np.where(df['Class 3'].isin(['price','invoice','bargain']), 'DAAN888PricingFinance@gmail.com', 'DAAN888Shipping@gmail.com'))),'')
        df['Classification5'] = np.where(df['Rating 4'] >= .75, np.where(df['Class 4'].isin(['expected','clean','quality','material','advertise','rating','size','workmanship']),'DAAN888ProductQuality@gmail.com', np.where(df['Class 4'].isin(['pleased','customer service','complain','contact','help','difficult','disclose','offensive']), 'DAAN888CustomerService@gmail.com',np.where(df['Class 4'].isin(['price','invoice','bargain']), 'DAAN888PricingFinance@gmail.com', 'DAAN888Shipping@gmail.com'))),'')
        #df['emailList'] = df[['Classification1','Classification2','Classification3','Classification4','Classification5']].apply(lambda x : ';'.join(np.unique(x)))
        #df['emailList'] = np.where(df['Classification1'] == df['Classification2'] or df['Classification1'] == df['Classification3'] or df['Classification1'] == df['Classification4'] or df['Classification1'] == df['Classification5'])
        st.dataframe(df)
        
        cols = ['Classification1','Classification2','Classification3','Classification4','Classification5']
        df['emailList'] = [[e for e in row if e==e and e != ""] for row in df[cols].values.tolist()]
        df['emailList'] = df['emailList'].apply(set)
        df['emailList'] = df['emailList'].apply(list)
        df['emailList'] = df['emailList'].apply(sorted)
        df['emailList'] = df['emailList'].apply(lambda x: "; ".join(map(str, x)))
        
        final_clean = final_clean[['cleanedEntry']]
        final = pd.merge(pd.merge(streamlitdf, final_clean,left_index=True, right_index=True),final_df,left_index=True, right_index=True)
        st.dataframe(final)
        final = final.rename(columns = {'cleanedEntry':'reviewText','originalEntry':'original_text'})
        csv_file = final.to_csv().encode('utf-8')
        csvfilename = 'SentimentClassificationOutput_'+str(datetime.now().strftime("%Y%m%d_%H%M%S"))+'.csv'
        
        
        
        
     
#Using a try except block to keep the download button hidden until the form has been ran
try:
    if final is not None:
        st.download_button(
        label="Download data as CSV",
        data=csv_file,
        file_name=csvfilename,
        mime='text/csv',
        )
        
        final.to_csv('Streamlitfiles\\SentimentClassificationOutput_'+str(datetime.now().strftime("%Y%m%d_%H%M%S"))+'.csv')
        full = pd.read_csv('combined_data.csv')
        full.drop('Unnamed: 0', axis=1, inplace=True)
        
        matchlogic = final[final['title'].isin(full['title']) & final['brand'].isin(full['brand']) & final['asin'].isin(full['asin']) & final['reviewTime'].isin(full['reviewTime']) & final['reviewText'].isin(full['reviewText']) & final['overall'].isin(full['overall']) & final['sub_category'].isin(full['sub_category'])]
        final.drop(matchlogic.index, inplace = True)
        
        if final.empty:
            st.write('All records already exist in PBI file')
        else:
            import email, smtplib, ssl
            from email import encoders
            from email.mime.base import MIMEBase
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            
            for x in final['emailList'].unique():
                emailattachment = final.loc[final['emailList'] == x]
                subject = "New Amazon Reviews For Review"+str(datetime.now().strftime("%Y%m%d_%H%M%S"))
                body = "This is an email with attachment sent from DAAN888 Team 8"
                sender_email = "Daan888team8@gmail.com"
                receiver_email = final['emailList'][0]
                password = 'DAAN888team8!'
                # Create a multipart message and set headers
                message = MIMEMultipart()
                message["From"] = sender_email
                message["To"] = receiver_email
                message["Subject"] = subject
                message["Bcc"] = receiver_email  # Recommended for mass emails
                
                # Add body to email
                message.attach(MIMEText(body, "plain"))
                
                def export_csv(df):
                  with io.StringIO() as buffer:
                    df.to_csv(buffer)
                    return buffer.getvalue()
                
                filename = 'reviews.csv'
                part = export_csv(emailattachment)
                
                # Encode file in ASCII characters to send by email    
                encoders.encode_base64(part)
                
                # Add header as key/value pair to attachment part
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {filename}",
                )
                
                # Add attachment to message and convert message to string
                message.attach(part)
                text = message.as_string()
                
                # Log in to server using secure context and send email
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, text)
            
            updated = pd.concat([full,final], axis=0, ignore_index=True)
            #Find a way to compress this if a record already exists based on that title, overall, and review text
            updated.to_csv('combined_data.csv')
            updated.to_excel('PowerBI\\combined_data.xlsx',engine = 'xlsxwriter')
        
except:
    pass
        

template = pd.DataFrame(columns=['title','brand','asin','reviewTime','reviewText','overall','sub_category'])
template = template.append({'title': 'OPTIONAL: This is the name of the item being reviewed', 'brand': 'OPTIONAL: This is the company that created the item being reviewed', 'sub_category':'OPTIONAL: This field is used for capturing the overall category of the item', 'asin': 'OPTIONAL: This is Amazons unique identifier value','reviewTime':'OPTIONAL: Time at which the review was submitted','reviewText':'REQUIRED: This is the main required item as it is the basis of the application','overall':'OPTIONAL: This is the star rating of the review with valid values of 1 through 5'}, ignore_index = True)
template = template.append({'title': 'Nike Men''s College Sideline Therma Pullover Hoodie', 'brand': 'Nike','sub_category':'Other', 'asin': 'B0BH88PDFK','reviewTime':'8/3/2022','reviewText':'Comfy and soft mens hoodie. Great fit and great design. It does snag easily but was great gift for my teenager.','overall':'5'}, ignore_index = True)
template = template.to_csv(index = False).encode('utf-8')

st.download_button(
    label = "Download Review Template",
    data = template,
    file_name = 'ReviewInputTemplate.csv',
    mime = 'text/csv',
    )

