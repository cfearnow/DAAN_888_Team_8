# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:21:08 2022

@author: cchee
"""

# Import necessary packages
import os
import glob
import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn import metrics
import json


# See current working directory
os.getcwd()


# Change working directory to parent directory (where your data is stored)

os.chdir('C:\\users\\cchee\\OneDrive\\Desktop')

#read the file


original_data = pd.read_parquet('new_presentiment_whole.parquet.gz', engine ='pyarrow')

original_sample = pd.read_csv ('final_data_set.csv')

del original_sample['sentiment']

new_data = pd.concat([original_data, original_sample])


#Check the size of the data

print(original_data.index)
print(original_sample.index)
print(new_data.index)

#remove the rows from the original sample

data_without_orgsample = new_data.drop_duplicates(subset=['idx'], keep=False)

print(data_without_orgsample.index)


# Create the new sample THIS IS WHERE I"M HAVING ISSUES


merge_trunc = data_without_orgsample.loc[(data_without_orgsample['reviewText'] >= 4) & (data_without_orgsample['reviewText'] < 512)]

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



def stratify_data(stratify_data, stratify_column_name, stratify_values, random_state=None):
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
    df_stratified = pd.DataFrame(columns = data_without_orgsample.columns) # Create an empty DataFrame with column names matching df_data

    for i in range(len(stratify_values)): # iterate over the stratify values (e.g. "Male, Undergraduate" etc.)

        df_filtered = data_without_orgsample[data_without_orgsample[stratify_column_name] == stratify_values[i]] # Filter the source data based on the currently selected stratify value
        df_temp = df_filtered.sample(replace=False, n=100, random_state=random_state) # Sample the filtered data using the calculated ratio
        
        df_stratified = pd.concat([df_stratified, df_temp]) # Add the sampled / stratified datasets together to produce the final result
        
    return df_stratified # Return the stratified, re-sampled data  

final = stratify_data(merge_trunc, 'Stratify', strat_list, random_state=42)

x = final.head(100)

z = final['Stratify'].value_counts()

final2 = final.drop(['Stratify', 'tokenized_num'], axis = 1)

final2.to_csv('final_data_set.csv', encoding='utf-8')
