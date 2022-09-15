# Import necessary packages
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import json
import gzip
import os
import codecs
import csv
import boto3
import io
import re
import string
import glob

# Packages for NLP
import spacy
from spacy.tokens import Span
from spacy import displacy

#%%
# Set directories

# Change working directory to parent directory (where your data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

#%%
# Create global variables for use in analysis

files = glob.glob('Data\Parquet\*.parquet')

#%%
# Load parquet files

# Function for loading data
def load_data(file):
    df = [pd.read_parquet(file, engine='pyarrow')]
    return df

# Load data into dictionary
def load_csvs(files_list):
    df_dict = {}
    for file in files_list:
        print('Loading: '+ file)
        df_dict[file] = load_data(file)
    return df_dict

df_dict = load_csvs(files)

print(df_dict.keys())

for i in df_dict.keys():
    j = i.split("\\", 2)[2]
    k = j.split(".",1)[0]
    print(k)

# Create single dataframe from all files including new column of sub_category
df_names = []
for i in df_dict.keys():
    j = i.split("\\", 2)[2]
    k = j.split(".",1)[0]
    temp_df = df_dict[i][0]
    temp_df['sub_category'] = k
    df_names.append(temp_df)

df_dict['merged'] = pd.concat(df_names)

all_file = df_dict['merged']

#all_file["sub_category"] = all_file['category'].str[0]

# Review head of dataframe to get an idea of the data we are dealing with
head_rows = all_file.head()

# Get rid of the category list field
all_file = all_file.drop('category', axis=1)

# Additional method of ingest

# Create single dataframe from dictionary if needed
Clothing_Shoes_and_Jewelry = df_dict['Data\Parquet\Clothing_Shoes_and_Jewelry.parquet'][0]
Office_Products = df_dict['Data\Parquet\Office_Products.parquet'][0]
Patio_Lawn_and_Garden = df_dict['Data\Parquet\Patio_Lawn_and_Garden.parquet'][0]
Pet_Supplies = df_dict['Data\Parquet\Pet_Supplies.parquet'][0]
Toys_and_Games = df_dict['Data\Parquet\Toys_and_Games.parquet'][0]

#%%
# Read individual files locally saved

#Name of file you want to read
review_file = 'Reviews\Home_and_Kitchen.json.gz' # Review data file path and name
meta_file = 'Metadata\meta_Toys_and_Games.json.gz' # Metadata file path and name
 
# Functions to read the data

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

#%%
# Run the functions
review_df = getDF(review_file) # create review data df
toys_meta_df = getDF(meta_file) # create metadata df

#%%
# Data exploration

# Make sure we only have 4 subcategories and nothing looks out of place
all_file['sub_category'].unique()

# Check the overall rating column
all_file['overall'].unique()

# Change the overall rating column to integer to get rid of the float variables
all_file['overall'] = all_file['overall'].astype(float)

# See data types of all columns
all_file.dtypes

# Calculate Total Reviews by sub_category
all_file['sub_category'].value_counts()

# Plot sub_category counts
all_file['sub_category'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Total Reviews", labelpad=14)
plt.ylabel("Sub Category", labelpad=14)
plt.title("Total Reviews by Sub Category", y=1.02);

# Function to calculate percentage null by category
def null_by_cat(file_name):
    nulls = []
    x = list(file_name.sub_category.unique())
    for i in x:
        y = file_name[file_name['sub_category'] == i].isnull().sum()
        z = len(file_name[file_name['sub_category'] == i])
        nulls.append([i, (y.reviewText/z)*100])
    return (nulls)

# Run percentage null function
null_values = pd.DataFrame(null_by_cat(all_file), columns = ['Sub_Category', 'Pct_Null'])

# Plot percentage null by sub category
null_values.plot.barh(x = 'Sub_Category', rot=0)
plt.xlabel("Percent", labelpad=14)
plt.ylabel("Sub Category", labelpad=14)
plt.title("Percent of Null Reviews by Sub Category", y=1.02);
     
# Create function to check for dulicate rows
def dups(file_name):
    x = file_name.duplicated()
    print(x.value_counts())
      
def dup_by_cat(file_name):
    x = file_name.duplicated()
    y = x.groupby('sub_category').value_counts()
    print(y)

check = dup_by_cat(all_file)

def clean_file(file_name):
    # Find number of null values in each dataset
    print('Below is the number of null values for each column in the dataset')
    # Check for null values
    print(file_name.isnull().sum())
    # Identify how many products have null reviews
    nulls_removed = file_name.dropna(subset=['reviewText'])
    print('Nulls have been removed. Current number of null values in the reviewTest field = ')
    # Check again for the number of null values
    print(nulls_removed['reviewText'].isnull().sum())      
    print('Checking for duplicate rows')
    # Check for duplicate values in each file
    dups(nulls_removed)
    print('Removing duplicate rows')
    # Remove duplicate rows and unnecessary columns from each file    
    dedup_reviews = nulls_removed.drop_duplicates()  
    print('Cleaning is complete')
    return (dedup_reviews)

# Text Cleaning Functions

def str_clean(file_name):
    # Remove line breaks
    file_name['reviewText'] = file_name['reviewText'].str.replace('(<br/>)', '')
    file_name['reviewText'] = file_name['reviewText'].str.replace('(<br>)', '')
    file_name['reviewText'] = file_name['reviewText'].str.replace('(</br>)', '')
    file_name['reviewText'] = file_name['reviewText'].str.replace('(<a).*(>).*(</a>)', '')
    # Remove ampersand
    file_name['reviewText'] = file_name['reviewText'].str.replace('(&amp)', '')
    # Remove greather than
    file_name['reviewText'] = file_name['reviewText'].str.replace('(&gt)', '')
    # Remove less than
    file_name['reviewText'] = file_name['reviewText'].str.replace('(&lt)', '')
    # Remove unicode hard space or a no-break space
    file_name['reviewText'] = file_name['reviewText'].str.replace('(\xa0)', ' ')
    return file_name

def txt_clean(file_name):
    # Make all strings lowercase
    file_name['Text'] = file_name['reviewText'].apply(lambda x: x.lower())
    # Remove any numerical digits
    file_name['Text'] = file_name['Text'].apply(lambda x: re.sub('\w*\d\w*','', x))
    # Remove all punctuation
    file_name['Text'] = file_name['Text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
    # Remove any extra spaces
    file_name['Text'] = file_name['Text'].apply(lambda x: re.sub(' +',' ',x))
    return(file_name)

All_Clean = clean_file(all_file)

All_Str_Clean = str_clean(All_Clean)

x = All_Str_Clean.head()

All_Str_Clean = txt_clean(All_Str_Clean)


############ END HERE ###############
    
#%%
# Start of NLP

# Create Spacy File (not sucessfully run yet)
nlp = spacy.load('en_core_web_sm')

docs = list(nlp.pipe(clean_reviews.reviewText))

print(docs)

def extract_tokens_plus_meta(doc:spacy.tokens.doc.Doc):
    """Extract tokens and metadata from individual spaCy doc."""
    return [
        (i.text, i.i, i.lemma_, i.ent_type_, i.tag_, 
         i.dep_, i.pos_, i.is_stop, i.is_alpha, 
         i.is_digit, i.is_punct) for i in doc
    ]

def tidy_tokens(docs):
    """Extract tokens and metadata from list of spaCy docs."""
    
    cols = [
        "doc_id", "token", "token_order", "lemma", 
        "ent_type", "tag", "dep", "pos", "is_stop", 
        "is_alpha", "is_digit", "is_punct"
    ]
    
    meta_df = []
    for ix, doc in enumerate(docs):
        meta = extract_tokens_plus_meta(doc)
        meta = pd.DataFrame(meta)
        meta.columns = cols[1:]
        meta = meta.assign(doc_id = ix).loc[:, cols]
        meta_df.append(meta)
        
    return pd.concat(meta_df) 

tidy_docs = tidy_tokens(docs)

tidy_docs.groupby("doc_id").size().hist(figsize=(14, 7), color="red", alpha=.4, bins=100);

tidy_docs.query("ent_type != ''").ent_type.value_counts()

tidy_docs.query("is_stop == False & is_punct == False").lemma.value_counts().head().plot(kind="barh", figsize=(24, 14), alpha=.7) 

str_cleaned = dedup_reviews
