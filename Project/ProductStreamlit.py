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
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import datetime

# Change working directory to parent directory (where your data is stored)
try:
    if str.split(os.getcwd(),"\\")[2] == "brull":
        os.chdir('C:\\Users\\brull\\OneDrive - The Pennsylvania State University\\Team-8\\Data\\final_data_set')
    else:
        os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data\\final_data_set')
except:
    os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data\\final_data_set')

brand = 'fisherprice'

st.title(str.capitalize(brand)+' Reviews')

# Load cleaned file
file = 'combined_data.csv'
#@st.cache
def load_data(brand):
    data = pd.read_csv(file)
    data = data[data['brand'] == brand]
    return data

count = st_autorefresh(interval=600000, limit=100, key="fizzbuzzcounter")

if count < 100:
    df = load_data(brand)
    df = df.drop(columns = ['OriginalIndex','Unnamed: 0'])
    #df = df.rename(columns = {'Unnamed: 0':'OriginalIndex'})

    df_random = df.sample(n=1000, random_state= 54321)
    st.subheader('Raw data')
    st.dataframe(df_random)

    st.bar_chart(data = df['Classification'].value_counts())
    st.line_chart(data = df['reviewTime'].value_counts())
    
    ct = datetime.datetime.now()
    st.write("last refresh at "+str(ct))

#Think about how to make this a production format
#scrape reviews for company
#
