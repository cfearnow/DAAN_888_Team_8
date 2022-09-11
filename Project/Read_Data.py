# Import necessary packages
import pandas as pd
import pyarrow.parquet as pq
import json
import gzip
import os
import codecs
import csv
import boto3
import io

# Packages for NLP
import spacy
from spacy.tokens import Span
from spacy import displacy

#%%
# See current working directory
os.getcwd()

# Change working directory to parent directory (where your data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project\Amazon_Data')

#%%
# Get Data from AWS S3

# Create bucket name and list of folders
bucket_name = 'amazon-reviews-pds'
folder = ['product_category=Office_Products', 'product_category=Pet_Products', 'product_category=Lawn_and_Garden', 'product_category=Toys'] 

# Create functions for downloading parquet files
def download_s3_parquet_file(s3, bucket, key):
    buffer = io.BytesIO()
    s3.Object(bucket, key).download_fileobj(buffer)
    return buffer

def load_data(folder = folder):
    client = boto3.client('s3')
    s3 = boto3.resource('s3')
    objects_dict = client.list_objects_v2(Bucket=bucket_name, Prefix = 'parquet/' + folder)
    s3_keys = [item['Key'] for item in objects_dict['Contents'] if item['Key'].endswith('.parquet')]
    buffers = [download_s3_parquet_file(s3, bucket_name, key) for key in s3_keys]
    dfs = [pq.read_table(buffer).to_pandas() for buffer in buffers]
    df = pd.concat(dfs, ignore_index=True)
    return(df)

# Read and create single dataframe for each folder named in 'folder' list   
gbl = globals()
for i in folder:
    gbl[i.split("=",1)[1]] = load_data(folder = i)

#%%
# Read individual files locally saved

#Name of file you want to read
review_file = 'Reviews\Home_and_Kitchen.json.gz' # Review data file path and name
meta_file = 'Metadata\meta_Home_and_Kitchen.json.gz' # Metadata file path and name
 
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
meta_df = getDF(meta_file) # create metadata df

#%%
# Data exploration

# Find number of null values in each dataset
review_df.isnull().sum()
meta_df.isnull().sum()

# Create function to check for dulicate rows
def dups(file_name):
    x = file_name.duplicated()
    print(x.value_counts())
    
# Review the column contents of each file for investigation
meta_head = meta_df.head()
review_head = review_df.head()

# Check for duplicate values in each file
dups(meta_df[['title','brand','main_cat','price','asin']])
dups(review_df[['overall','verified','reviewTime','reviewerName','reviewText','summary','asin']])

# Remove duplicate rows and unnecessary columns from each file
dedup_meta = meta_df[['title','brand','main_cat','price','asin']].drop_duplicates()
dedup_reviews = review_df[['overall','verified','reviewTime','reviewerName','reviewText','summary','asin']].drop_duplicates()
 
# View unique product titles
unique_names = meta_df['title'].unique()

# Merge the data
productreviews = pd.merge(dedup_meta, dedup_reviews, on='asin', how='inner')

# Identify how many products have null reviews
productreviews.isnull().sum()

# Drop all rows if "reviewText" 
nonull_reviews = productreviews.dropna(subset=['reviewText'])

# Ensure nulls were removed
nonull_reviews.isnull().sum()

# Check one last time to ensure there are no duplicate rows
print(nonull_reviews.duplicated().value_counts())

# Make copy before text changes are done
clean_reviews = nonull_reviews.copy()
#maybe = dedup_reviews

#%%
# Text Cleaning

# Create function to modify text
def preprocess(ReviewText):
    # Remove line breaks
    ReviewText = ReviewText.str.replace('(<br/>)', '')
    
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    # Remove ampersand
    ReviewText = ReviewText.str.replace('(&amp)', '')
    # Remove greather than
    ReviewText = ReviewText.str.replace('(&gt)', '')
    # Remove less than
    ReviewText = ReviewText.str.replace('(&lt)', '')
    # Remove unicode hard space or a no-break space
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')  
    return ReviewText

# Run function
clean_reviews['reviewText'] = preprocess(clean_reviews['reviewText'])

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
