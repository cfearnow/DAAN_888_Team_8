# Import necessary packages
import pandas as pd
import json
import gzip
import os
import time

start = time.perf_counter()

# See current working directory
os.getcwd()

# Change working directory to parent directory (where your data is stored)
try:
    if str.split(os.getcwd(),"\\")[2] == "brull":
        os.chdir('C:\\Users\\brull\\OneDrive - The Pennsylvania State University\\Team-8\\Data')
    else:
        os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data')
except:
    os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data')

#%%
# Read individual files

#Name of file you want to read
review_file = 'Reviews\Sports_and_Outdoors.json.gz' # Review data file path and name
meta_file = 'Metadata\meta_Sports_and_Outdoors.json.gz' # Metadata file path and name

try:
    outpath = '.\\Output\\'+str.split(str.split(review_file,"\\")[1],".")[0]
except:
    outpath = '.\\Output\\'+review_file
 
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


productreviews = pd.merge(meta_df[['category', 'title', 'brand', 'main_cat', 'price', 'asin']], 
                          review_df[['overall','verified','reviewTime','reviewerName','reviewText','summary','asin']], 
                          on='asin', how='inner')

productreviews['category'].astype(str)
productreviews['title'].astype(str)
productreviews['brand'].astype(str)
productreviews['main_cat'].astype(str)
productreviews['price'].astype(str)
productreviews['asin'].astype(str)
productreviews['overall'].astype(str)
productreviews['reviewTime'].astype(str)
productreviews['reviewerName'].astype(str)
productreviews['reviewText'].astype(str)
productreviews['summary'].astype(str)

print(productreviews.head())

import pyarrow
productreviews.to_parquet(outpath+"v2.parquet",partition_cols='overall' , engine='pyarrow')

end = time.perf_counter()
print(f"Code finished in {(end - start)/60:0.4f} minutes")
