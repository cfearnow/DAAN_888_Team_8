# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 18:15:15 2022

@author: brull
"""

import pyarrow
import pandas as pd
import os
import numpy as np
import time

#timer to see how long code is taking from start to finish
start = time.perf_counter()

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
pets = pd.read_parquet("Pet_Supplies.parquet", engine='pyarrow')
office = pd.read_parquet("Office_Products.parquet", engine='pyarrow')
lawn = pd.read_parquet("Lawn_and_Garden.parquet", engine='pyarrow')

#combine datasets into 1
merged1 = toys.append(pets, ignore_index=True)
merged2 = merged1.append(office, ignore_index=True)
merged = merged2.append(lawn, ignore_index=True)

#Convert price column to usable float format
merged['price'] = merged['price'].apply(lambda x: x if len(x) < 25 else "$0.00")
merged['price'] = merged['price'].apply(lambda x: x if x != '\n\n\n<script' else "$0.00")
merged['price'] = merged['price'].replace('','$0.00').replace({'\,':''}, regex = True).replace({'\$':''}, regex = True).str.split(' - ')
merged['price'] = merged['price'].apply(lambda x: np.array(x, dtype=np.float32))
merged['price'] = merged['price'].map(lambda x: x.mean())
merged['price'] = merged['price'].apply(lambda x: x if x > 0.00 else '') 

#convert review time date format mm d, yyyy to yyyy/mm/dd
from datetime import datetime
merged['reviewTime'].apply(lambda x: datetime.strptime(x,"%m %d, %Y").strftime("%Y/%m/%d"))

end = time.perf_counter()
print(f"Code finished in {(end - start)/60:0.4f} minutes")