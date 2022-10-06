# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import gzip
import os
import re
import string
import glob

#%%
# Set directories

# Change working directory to parent directory (where your data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

#%%
# Create global variables for use in analysis

files = glob.glob('Data\Parquet\*.parquet')

#%%
# Load parquet files and read all at once or individually

# Function for loading data
def load_data(file):
    df = [pd.read_parquet(file, engine='pyarrow')]
    return df

# Load all data into dictionary
def load_csvs(files_list):
    df_dict = {}
    for file in files_list:
        # Skip Clothing_Shoes_and_Jewelry file
        if file == 'Data\Parquet\Clothing_Shoes_and_Jewelry.parquet':
            continue
  
        print('Loading: '+ file)
        df_dict[file] = load_data(file)
    return df_dict

df_dict = load_csvs(files)

# Create single dataframe of all files including new column of sub_category
df_names = []
for i in df_dict.keys():
    j = i.split("\\", 2)[2]
    k = j.split(".",1)[0]
    temp_df = df_dict[i][0]
    temp_df['sub_category'] = k
    df_names.append(temp_df)

df_dict['merged'] = pd.concat(df_names)

# Create the file in pandas df
all_file = df_dict['merged']

# Review head of dataframe to get an idea of the data we are dealing with
head_rows = all_file.head()

# Get rid of the category list field
all_file = all_file.drop('category', axis=1)

#%%
# Individual file method of ingest

# Create single dataframe from dictionary if needed
### Not Used ### Clothing_Shoes_and_Jewelry = df_dict['Data\Parquet\Clothing_Shoes_and_Jewelry.parquet'][0]
Office_Products = df_dict['Data\Parquet\Office_Products.parquet'][0]
Patio_Lawn_and_Garden = df_dict['Data\Parquet\Patio_Lawn_and_Garden.parquet'][0]
Pet_Supplies = df_dict['Data\Parquet\Pet_Supplies.parquet'][0]
Toys_and_Games = df_dict['Data\Parquet\Toys_and_Games.parquet'][0]

#%%
# Read non-parquet files that are locally saved

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

# Run the functions
review_df = getDF(review_file) # create review data df
toys_meta_df = getDF(meta_file) # create metadata df

#%%
# Data exploration

# Make sure we only have 4 subcategories and nothing looks out of place
all_file['sub_category'].unique()

# See data types of all columns
all_file.dtypes

# Check the overall rating column
all_file['overall'].unique()

# Change the overall rating column to float
all_file['overall'] = all_file['overall'].astype(float)

# Adjust the price field to float
def product_price(file_name, existing_column, new_column):
    file_name[new_column] = file_name[existing_column].copy()
    file_name[new_column] = file_name[new_column].apply(lambda x: x if len(x) < 25 else "$0.00")
    file_name[new_column] = file_name[new_column].apply(lambda x: x if x != '\n\n\n<script' else "$0.00")
    file_name[new_column] = file_name[new_column]\
        .replace('','$0.00').replace({'\,':''}, regex = True)\
        .replace({'\$':''}, regex = True)\
        .str.split(' - ')
    #file_name[new_column] = pd.to_numeric(file_name[new_column], errors='coerce')
    file_name[new_column] = file_name[new_column].apply(lambda x: np.array(x, dtype=np.float32))
    file_name[new_column] = file_name[new_column].map(lambda x: x.mean())
    #file_name[new_column] = file_name[new_column].apply(lambda x: x if x > 0.00 else '')
    return(file_name)

all_file = product_price(all_file, 'price', 'price_adj')

# Change the reviewTime field to datetime
all_file['reviewTime'] = all_file['reviewTime'].apply(lambda x: datetime.strptime(x,"%m %d, %Y").strftime("%Y/%m/%d"))

# See data types of all columns one more time to ensure changes were successful
all_file.dtypes

# Calculate Total Reviews by sub_category
all_file['sub_category'].value_counts()

# Plot sub_category counts
a = pd.DataFrame(all_file['sub_category'].value_counts().sort_values(ascending=True))
x = list(a.sub_category)
y = list(a.index)
plt.barh(y, x, color='maroon')
plt.xlabel("Total Reviews (M)")
plt.ylabel("Sub Category")
plt.title("Total Reviews by Sub Category")
current_values = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:1.0f}M'.format(x*1e-6) for x in current_values])
for y, x in enumerate(a['sub_category']):
    plt.annotate(str(round((int(x)/1000000),1)) + 'M', xy=(x, y), va='center')
plt.show()

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
null_values = pd.DataFrame(null_by_cat(all_file), columns = ['sub_category', 'pct_null'])

# Plot percentage null by sub category
x = list(null_values.pct_null)
y = list(null_values.sub_category)
plt.barh(y, x)
plt.xlabel("Pct Null Reviews")
plt.ylabel("Sub Category")
plt.title("Percent of Null Reviews by Sub Category")
current_values = plt.gca().get_xticks()
for y, x in enumerate(null_values['pct_null']):
    plt.annotate(str(round(float(x),2)) + '%', xy=(x, y), va='center')
plt.show()

# Plot number of reviews by subcategory and rating
pivot = pd.pivot_table(all_file, values = 'asin', index = 'sub_category', columns = 'overall', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'sub_category', y = [1,2,3,4,5], rot = 50)
plt.xlabel("Sub Category")
plt.ylabel("Number of Reviews")
plt.title("Reviews by Sub Category and Rating (millions)")
plt.legend(title="Rating", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:1.0f}M'.format(y*1e-6) for y in current_values])
plt.show()
  
# Create function to check for dulicate rows
def dups(file_name):
    x = file_name.duplicated()
    print(x.value_counts())
      
# Plot number of duplicate values by subcategory 
all_file['dups'] = all_file.duplicated()
all_dups = pd.DataFrame(all_file.value_counts(subset=['sub_category', 'dups']).sort_index()).reset_index()
all_dups.columns = ['sub_category', 'duplicate', 'count']
dups_pivot = all_dups.pivot_table('count', 'sub_category', 'duplicate').reset_index()
dups_pivot.columns = ['sub_category', 'No', 'Yes']

dups_pivot.plot.bar(x = 'sub_category',y = ['No', 'Yes'], rot = 50)
plt.xlabel("Sub Category")
plt.ylabel("Number of Reviews")
plt.title("Duplicates by Sub Category and Rating (millions)")
plt.legend(title="Duplicate", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:1.0f}M'.format(y*1e-6) for y in current_values])
plt.show()

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

def str_clean(file_name, file_column):
    # Remove line breaks
    file_name[file_column] = file_name[file_column].str.replace('(<br/>)', '')
    file_name[file_column] = file_name[file_column].str.replace('(<br>)', '')
    file_name[file_column] = file_name[file_column].str.replace('(</br>)', '')
    # Remove additional line breaks
    file_name[file_column] = file_name[file_column].str.replace('\n',' ')
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

all_clean = clean_file(all_file)

# Get rid of the reviewerName field
all_clean = all_clean.drop('reviewerName', axis=1)

# Run the str_clean function on necessary fields
all_clean_adj = str_clean(all_clean, 'title')
all_clean_adj = str_clean(all_clean_adj, 'brand')
all_clean_adj = str_clean(all_clean_adj, 'main_cat')
all_clean_adj = str_clean(all_clean_adj, 'reviewText')
all_clean_adj = str_clean(all_clean_adj, 'summary')

all_clean_adj['original_text'] = all_clean_adj['reviewText'].copy()
all_clean_adj['original_summary'] = all_clean_adj['summary'].copy()


final_clean = txt_clean(all_clean_adj, 'title')
final_clean = txt_clean(final_clean, 'brand')
final_clean = txt_clean(final_clean, 'main_cat')
final_clean = txt_clean(final_clean, 'reviewText')
final_clean = txt_clean(final_clean, 'summary')

def two_clean(file_name, file_column):
    # Remove all words with lenght less than 2
    file_name[file_column] = file_name[file_column].apply(lambda y: ' '.join([w for w in str(y).split() if len(w)>2]))
    return(file_name)

final_clean_2 = two_clean(final_clean, 'reviewText')
final_clean_2 = two_clean(final_clean_2, 'summary')

# Create a new index column to ensure no duplicates
final_clean_2['idx'] = range(1, len(final_clean_2) + 1)

# Extract only review columns to assist with file size

orig_cols = [
'title',
'brand',
'main_cat',
'price',
'asin',
'verified',
'reviewTime',
'reviewText',
'summary',
'overall',
'sub_category',
'price_adj']
    
def orig_dups(file_name):
    x = file_name[file_name.duplicated(orig_cols)]
    print(x.value_counts())
    
def clean_file2(file_name):
    print('Checking for duplicate rows')
    # Check for duplicate values in each file
    orig_dups(file_name)
    print('Removing duplicate rows')
    # Remove duplicate rows and unnecessary columns from each file    
    dedup_reviews = file_name.drop_duplicates(subset = orig_cols)  
    print('Cleaning is complete')
    return (dedup_reviews)
    
x = clean_file2(final_clean_2)

x.to_parquet('Newest_brand_new_file_clean.parquet.gz', compression='gzip')

############ END HERE ###############