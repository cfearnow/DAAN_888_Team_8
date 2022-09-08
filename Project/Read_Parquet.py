# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:57:27 2022

@author: brull
"""

import pyarrow
import pandas as pd
import os

# See current working directory
os.getcwd()

# Change working directory to parent directory (where your data is stored)
try:
    if str.split(os.getcwd(),"\\")[2] == "brull":
        os.chdir('C:\\Users\\brull\\OneDrive - The Pennsylvania State University\\Team-8\\Data\\Output')
    else:
        os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data')
except:
    os.chdir('D:\\School_Files\\DAAN_888\\Team_8_Project\\Amazon_Data')

toys = pd.read_parquet("Toys_and_Games.parquet", engine='pyarrow')

toys['category'] = toys['category'].astype(str)

pd.options.display.max_columns = None
print(toys.head())

print(toys.isnull().sum())
print(len(toys))

print("Category")
#print(toys.category.unique())
print(len(pd.unique(toys['category'])))
print("Title")
#print(toys.title.unique())
print(len(pd.unique(toys['title'])))
print("Brand")
#print(toys.brand.unique())
print(len(pd.unique(toys['brand'])))
print("Main Category")
#print(toys.main_cat.unique())
print(len(pd.unique(toys['main_cat'])))
print("Price")
#print(toys.price.unique())
print(len(pd.unique(toys['price'])))
print("Rating")
#print(toys.overall.unique())
print(len(pd.unique(toys['overall'])))
print("Verified")
#print(toys.verified.unique())
print(len(pd.unique(toys['verified'])))
print("Review Time")
#print(toys.reviewTime.unique())
print(len(pd.unique(toys['reviewTime'])))
print("Reviewer")
#print(toys.reviewerName.unique())
print(len(pd.unique(toys['reviewerName'])))
print("Review Text")
#print(toys.reviewText.unique())
print(len(pd.unique(toys['brand'])))
print("Review Header")
#print(toys.summary.unique())
print(len(pd.unique(toys['summary'])))