# -*- coding: utf-8 -*-
# Import packages 
import pandas as pd
import json
import gzip
import os
import re
import string


# See current working directory
os.getcwd()


# Change working directory to parent directory (where your data is stored)

os.chdir('C:\\users\\cchee\\OneDrive\\Desktop')

#read the files
review_file = 'Toys_and_Games.json.gz' # Review data file path and name
meta_file = 'meta_Toys_and_Games.json.gz' # Metadata file path and name

# Parse through the json file
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

# Convert to pandas df
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

review_df = getDF(review_file) # create review data df
meta_df = getDF(meta_file) # create metadata df

# Find number of null values in each dataset
review_df.isnull().sum()

# Find number of null values in each dataset
meta_df.isnull().sum()

# Merge review and meta data

productreviews = pd.merge(meta_df[['title','brand','main_cat','price','asin']], review_df[['overall','verified','reviewTime','reviewerName','reviewText','summary','asin']], on='asin', how='inner')
print(productreviews.head())

#Check for missing values

productreviews.isnull().sum()

# remove rows with missing reviews and summaries

nonull_reviews = productreviews.dropna(subset=['reviewText', 'summary'])
nonull_reviews.isnull().sum()
