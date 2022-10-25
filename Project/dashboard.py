# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:08:02 2022

@author: chris
"""

# Import necessary packages
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import os
import glob
import seaborn as sns

from sklearn import metrics

# Set directories

# Change working directory to parent directory (where data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

# Directory to chunked results
data = pd.read_csv('classified_file.csv')

class_list = ['0', '1', '2', '3', '4']

# Plot number of reviews by subcategory and rating

for num in class_list:
    target = 'Class ' + num
    text_pivot = pd.pivot_table(data, values = 'title', index = target, aggfunc= 'count')
    text_pivot = text_pivot.reset_index()
    text_pivot.plot.bar(x = target, y = 'title', rot = 90)
    plt.xlabel("Category")
    plt.ylabel("Total Count")
    plt.title("Classification by Category")
    current_values = plt.gca().get_yticks()
    plt.show()
    
    
count_pivot = pd.pivot_table(data, values = 'title', index = target, aggfunc= 'count')

data['Class 4'].sum()

# iterating the columns
for col in data.columns:
    print(col)

#reshape DataFrame from wide format to long format
long_1 = data.drop(['Class 1', 'Rating 1', 'Class 2', 'Rating 2', 'Class 3', 'Rating 3', 'Class 4', 'Rating 4'], axis = 1)
long_1.rename(columns={'Class 0': 'Class', 'Rating 0': 'Rating'}, inplace=True)
long_1['Run'] = 1

long_2 = data.drop(['Class 0', 'Rating 0', 'Class 2', 'Rating 2', 'Class 3', 'Rating 3', 'Class 4', 'Rating 4'], axis = 1)
long_2.rename(columns={'Class 1': 'Class', 'Rating 1': 'Rating'}, inplace=True)
long_2['Run'] = 2

long_3 = data.drop(['Class 0', 'Rating 0', 'Class 1', 'Rating 1', 'Class 3', 'Rating 3', 'Class 4', 'Rating 4'], axis = 1)
long_3.rename(columns={'Class 2': 'Class', 'Rating 2': 'Rating'}, inplace=True)
long_3['Run'] = 3

long_4 = data.drop(['Class 0', 'Rating 0', 'Class 1', 'Rating 1', 'Class 2', 'Rating 2', 'Class 4', 'Rating 4'], axis = 1)
long_4.rename(columns={'Class 3': 'Class', 'Rating 3': 'Rating'}, inplace=True)
long_4['Run'] = 4

long_5 = data.drop(['Class 0', 'Rating 0', 'Class 1', 'Rating 1', 'Class 2', 'Rating 2', 'Class 3', 'Rating 3'], axis = 1)
long_5.rename(columns={'Class 4': 'Class', 'Rating 4': 'Rating'}, inplace=True)
long_5['Run'] = 5

frames = [long_1, long_2, long_3, long_4, long_5]

long_data = pd.concat(frames)

#def plot3D(result, wordgroups):

    
axes = [0, 1, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result[:, axes[0]], result[:, axes[1]], result[:, axes[2]])
for g, group in enumerate(wordgroups):
    for word in group:
        if not word in words:
            continue
        i = words.index(word)
        # Create plot point
        color = colors[g] if g < len(colors) else defaultcolor
        size = sizes[g] if g < len(sizes) else defaultsize
        ax.text(result[i, axes[0]], result[i, axes[1]],
                result[i, axes[2]], word, color=color, fontsize=size)

