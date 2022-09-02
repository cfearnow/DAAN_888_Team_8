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

