# Import necessary packages
import pandas as pd
import json
import gzip
import os

# See current working directory
os.getcwd()

# Change working directory to parent directory (where your data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project\Amazon_Data')

#%%
# Read individual files

#Name of file you want to read
review_file = 'Reviews\Toys_and_Games.json.gz' # Review data file path and name
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
meta_df = getDF(meta_file) # create metadata df


# Alex already created parquet file for Toys_and_games

# Clothing_Shoes_and_Jewelry.json.gz parquet file creation

# Load Data
review_file = 'Clothing_Shoes_and_Jewelry.json.gz' # Review data file path and name
meta_file = 'meta_Clothing_Shoes_and_Jewelry.json.gz' # Metadata file path and name

# Run functions to parse and get dataframe
review_df = getDF(review_file) # create review data df
meta_df = getDF(meta_file) # create metadata df

# Confirm field names
review_df.iloc[0]

# Create merged file
productreviews = pd.merge(meta_df[['category','title','brand','main_cat','price','asin']], review_df[['overall','verified','reviewTime','reviewerName','reviewText','summary','asin']], on='asin', how='inner')

# Output to parquet
productreviews.to_parquet('Clothing_Shoes_and_Jewelry.parquet',partition_cols=['overall'])


# Office_Products.json.gz parquet file creation

# Load Data
review_file = 'Office_Products.json.gz' # Review data file path and name
meta_file = 'meta_Office_Products.json.gz' # Metadata file path and name

# Run functions to parse and get dataframe
review_df = getDF(review_file) # create review data df
meta_df = getDF(meta_file) # create metadata df

# Confirm field names
review_df.iloc[0]

# Create merged file
productreviews = pd.merge(meta_df[['category','title','brand','main_cat','price','asin']], review_df[['overall','verified','reviewTime','reviewerName','reviewText','summary','asin']], on='asin', how='inner')

# Output to parquet
productreviews.to_parquet('Office_Products.parquet',partition_cols=['overall'])


# Patio_Lawn_and_Garden.json.gz parquet file creation

# Load Data
review_file = 'Patio_Lawn_and_Garden.json.gz' # Review data file path and name
meta_file = 'meta_Patio_Lawn_and_Garden.json.gz' # Metadata file path and name

# Run functions to parse and get dataframe
review_df = getDF(review_file) # create review data df
meta_df = getDF(meta_file) # create metadata df

# Confirm field names
review_df.iloc[0]

# Create merged file
productreviews = pd.merge(meta_df[['category','title','brand','main_cat','price','asin']], review_df[['overall','verified','reviewTime','reviewerName','reviewText','summary','asin']], on='asin', how='inner')

# Output to parquet
productreviews.to_parquet('Patio_Lawn_and_Garden.parquet',partition_cols=['overall'])


# Pet_Supplies.json.gz parquet file creation

# Load Data
review_file = 'Pet_Supplies.json.gz' # Review data file path and name
meta_file = 'meta_Pet_Supplies.json.gz' # Metadata file path and name

# Run functions to parse and get dataframe
review_df = getDF(review_file) # create review data df
meta_df = getDF(meta_file) # create metadata df

# Confirm field names
review_df.iloc[0]

# Create merged file
productreviews = pd.merge(meta_df[['category','title','brand','main_cat','price','asin']], review_df[['overall','verified','reviewTime','reviewerName','reviewText','summary','asin']], on='asin', how='inner')

# Output to parquet
productreviews.to_parquet('Pet_Supplies.parquet',partition_cols=['overall'])





