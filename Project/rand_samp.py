# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:01:55 2022

@author: chris
"""
# Import necessary packages
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn import metrics

#%%
# Set directories

# Change working directory to parent directory (where your data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

# Directory to chunked results
chunked_path = './Data/chunked'

#%%
# Create global variables for use in analysis
files = glob.glob('Data\Sent_Analysis\Clean_Data\All_Cols\*.parquet.gz')

clean = pd.read_parquet(files, engine='pyarrow')

# Create a new index column to ensure no duplicates
clean['idx'] = range(1, len(clean) + 1)

#clean_2 = clean[['idx', 'original_text']]

#%%
# Read and append all chunked data

chunk_files = glob.glob(os.path.join(chunked_path, "*.csv"))

chunk_df = pd.concat((pd.read_csv(f) for f in chunk_files), ignore_index=True)

chunk_df = chunk_df.drop(['Unnamed: 0', 'original_text'], axis = 1)

clean_merge = pd.merge(clean, chunk_df, how = 'inner', on = ['idx'])

merge_trunc = clean_merge.loc[(clean_merge['tokenized_num'] >= 4) & (clean_merge['tokenized_num'] < 512)]

merge_trunc['overall'] = merge_trunc['overall'].astype(int)

merge_trunc = merge_trunc[(merge_trunc['reviewText'].notnull()) & (merge_trunc['reviewText']!=u'')]

orig_conditions = [
    (merge_trunc['overall'] == 5) | (merge_trunc['overall'] == 4),
    (merge_trunc['overall'] == 2) | (merge_trunc['overall'] == 1),
    (merge_trunc['overall'] == 3)
    ]

values = ['Positive', 'Negitive', 'Neutral']

merge_trunc['sentiment'] = np.select(orig_conditions, values)

merge_trunc['Stratify'] = merge_trunc['sentiment'] + ", " + merge_trunc['sub_category']

y = merge_trunc['Stratify'].value_counts()

strat_list = ['Positive, Toys_and_Games',
              'Positive, Pet_Supplies',
              'Positive, Office_Products',
              'Positive, Patio_Lawn_and_Garden',
              'Negitive, Toys_and_Games',
              'Negitive, Pet_Supplies',
              'Negitive, Patio_Lawn_and_Garden',
              'Negitive, Office_Products',
              #'Neutral, Toys_and_Games',
              #'Neutral, Pet_Supplies',
              #'Neutral, Office_Products',
              #'Neutral, Patio_Lawn_and_Garden'
              ]

#%%
def stratify_data(df_data, stratify_column_name, stratify_values, random_state=None):
    """Stratifies data according to the values and proportions passed in
    Args:
        df_data (DataFrame): source data
        stratify_column_name (str): The name of the single column in the dataframe that holds the data values that will be used to stratify the data
        stratify_values (list of str): A list of all of the potential values for stratifying e.g. "Male, Graduate", "Male, Undergraduate", "Female, Graduate", "Female, Undergraduate"
        stratify_proportions (list of float): A list of numbers representing the desired propotions for stratifying e.g. 0.4, 0.4, 0.2, 0.2, The list values must add up to 1 and must match the number of values in stratify_values
        random_state (int, optional): sets the random_state. Defaults to None.
    Returns:
        DataFrame: a new dataframe based on df_data that has the new proportions represnting the desired strategy for stratifying
    """
    df_stratified = pd.DataFrame(columns = df_data.columns) # Create an empty DataFrame with column names matching df_data

    for i in range(len(stratify_values)): # iterate over the stratify values (e.g. "Male, Undergraduate" etc.)

        df_filtered = df_data[df_data[stratify_column_name] == stratify_values[i]] # Filter the source data based on the currently selected stratify value
        df_temp = df_filtered.sample(replace=False, n=50000, random_state=random_state) # Sample the filtered data using the calculated ratio
        
        df_stratified = pd.concat([df_stratified, df_temp]) # Add the sampled / stratified datasets together to produce the final result
        
    return df_stratified # Return the stratified, re-sampled data  

final = stratify_data(merge_trunc, 'Stratify', strat_list, random_state=42)

x = final.head(100)

z = final['Stratify'].value_counts()

final2 = final.drop(['Stratify', 'tokenized_num'], axis = 1)

final2.to_csv('final_data_set.csv', encoding='utf-8')

#%%
def chunk_data(file_name):
    
    n = 100
    chunk_reviews = np.array_split(file_name, n)
    
    # Check if augmented directory exists
    folder_check = os.path.isdir(chunked_path)

    if not folder_check:
        os.makedirs(chunked_path)
        print("created folder : ", chunked_path)
    else:
        print(chunked_path, "already exists.")
        
    for i in range(len(chunk_reviews)):
        varname = 'chunk_' + str(i) 
        new_name = varname + '.csv'
        folder_path = os.path.join(chunked_path, new_name)
        exists_check = os.path.isdir(folder_path)
        
        if not exists_check:
            varname = chunk_reviews[i]
            varname['tokenized_num'] = varname.apply(lambda row: word_tokenize(row['original_text']), axis=1).str.len()
            varname.to_csv(folder_path, encoding='utf-8')
            print("created file : ", new_name)
        else:
            print(new_name, " already exists.")

chunk_data(clean_2)

#%%

final = pd.read_csv('final_data_set.csv')

sid_analyzer = SentimentIntensityAnalyzer()
#############################################

def get_sentiment(text:str, analyser,desired_type:str='pos'):
    # Get sentiment from text
    sentiment_score = analyser.polarity_scores(text)
    return sentiment_score[desired_type]

# Get Sentiment scores
def get_sentiment_scores(df,data_column):
    df[f'{data_column} Positive Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'pos'))
    df[f'{data_column} Negative Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neg'))
    df[f'{data_column} Neutral Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neu'))
    df[f'{data_column} Compound Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'compound'))
    return df

get_sentiment_scores(final, 'reviewText')

###############################################################################

text_conditions = [
    (final['reviewText Compound Sentiment Score'] > 0.5),
    (final['reviewText Compound Sentiment Score'] < -0.5),
    (final['reviewText Compound Sentiment Score'] <= 0.5) & (final['reviewText Compound Sentiment Score'] >= -0.5)
]

values = ['Positive', 'Negitive', 'Neutral']

final['text_sentiment'] = np.select(text_conditions, values)  

Toys_Games = final[final['sub_category'] == 'Toys_and_Games']
Office_Products = final[final['sub_category'] == 'Office_Products']
Pet_Supplies = final[final['sub_category'] == 'Pet_Supplies']
Patio_Lawn_Garden = final[final['sub_category'] == 'Patio_Lawn_and_Garden']



chart_list = [Toys_Games, Office_Products, Pet_Supplies, Patio_Lawn_Garden]


final.to_csv('All_Cats.csv', encoding='utf-8')
Toys_Games.to_csv('Toys_Games.csv', encoding='utf-8')
Office_Products.to_csv('Office_Products.csv', encoding='utf-8')
Pet_Supplies.to_csv('Pet_Supplies.csv', encoding='utf-8')
Patio_Lawn_Garden.to_csv('Patio_Lawn_Garden.csv', encoding='utf-8')

# Plot number of reviews by subcategory and rating

    text_pivot = pd.pivot_table(i, values = 'title', index = 'overall', columns = 'text_sentiment', aggfunc= 'count')
    text_pivot = text_pivot.reset_index()
    text_pivot.plot.bar(x = 'overall', y = ['Negitive', 'Neutral', 'Positive'], rot = 50)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.title("Text Reviews by Overall Star Rating and Sentiment")
    plt.legend(title="Rating", loc='upper left', fontsize='small', fancybox=True)
    current_values = plt.gca().get_yticks()
    plt.show()