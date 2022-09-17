# Import necessary packages
import pandas as pd
import json
import gzip
import os
import time

#timer to see how long code is taking from start to finish
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

#Name of file you want to read
review_file = 'Reviews\Office_Products.json.gz' # Review data file path and name
meta_file = 'Metadata\meta_Office_Products.json.gz' # Metadata file path and name
 
try:
    outpath = '.\\Output\\'+str.split(str.split(review_file,"\\")[1],".")[0]+'.parquet'
except:
    outpath = '.\\Output\\'+review_file+'.parquet'
    
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

meta_df = meta_df[['title','brand','main_cat','price','asin']]
review_df = review_df[['overall','verified','reviewTime','reviewerName','reviewText','summary','asin']]

#drop duplicate values from each dataset prior to merge
dedup_meta = meta_df.drop_duplicates()
dedup_reviews = review_df.drop_duplicates()

productreviews = pd.merge(dedup_meta, dedup_reviews, on='asin', how='inner')

nonull_reviews = productreviews.dropna(subset=['reviewText'])

# Output to parquet
nonull_reviews.to_parquet(outpath,partition_cols=['overall'])

end = time.perf_counter()
print(f"Code finished in {(end - start)/60:0.4f} minutes")