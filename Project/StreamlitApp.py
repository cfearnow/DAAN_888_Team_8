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

st.title('Test Amazon Reviews')

#count = st_autorefresh(interval=60000, limit=100, key="fizzbuzzcounter")

# Load cleaned file
file = 'test.csv'
#@st.cache
def load_data(nrows):
    data = pd.read_csv(file, nrows = nrows)
    return data

count = st_autorefresh(interval=900000, limit=100, key="fizzbuzzcounter")

if count < 100:
    df = load_data(400000)
    
    df_random = df#.sample(n=1000, random_state= 54321)
    st.subheader('Raw data')
    st.dataframe(df_random)
    
    ct = datetime.datetime.now()
    st.write("last refresh at "+str(ct))

# if count == 0:
#     df = load_data(400000)
    
#     df_random = df#.sample(n=1000, random_state= 54321)
#     st.subheader('Raw data')
#     st.dataframe(df_random)
    
#     ct = datetime.datetime.now()
#     st.write("last refresh at "+str(ct))

# elif count % 3 == 0 and count % 5 == 0:
#     df = load_data(400000)
    
#     df_random = df#.sample(n=1000, random_state= 54321)
#     st.subheader('Raw data')
#     st.dataframe(df_random)
    
#     ct = datetime.datetime.now()
#     st.write("last refresh at "+str(ct))
    
# elif count % 3 == 0:
#     df = load_data(400000)
    
#     df_random = df#.sample(n=1000, random_state= 54321)
#     st.subheader('Raw data')
#     st.dataframe(df_random)
    
#     ct = datetime.datetime.now()
#     st.write("last refresh at "+str(ct))
    
# elif count % 5 == 0:
#     df = load_data(400000)
    
#     df_random = df#.sample(n=1000, random_state= 54321)
#     st.subheader('Raw data')
#     st.dataframe(df_random)
    
#     ct = datetime.datetime.now()
#     st.write("last refresh at "+str(ct))
    
# else:
#     df = load_data(400000)
    
#     df_random = df#.sample(n=1000, random_state= 54321)
#     st.subheader('Raw data')
#     st.dataframe(df_random)
    
#     ct = datetime.datetime.now()
#     st.write("last refresh at "+str(ct))





#Think about how to make this a production format
#scrape reviews for company
#
