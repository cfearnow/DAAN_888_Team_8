# Import necessary packages
import pandas as pd
import numpy as np
from sklearn import metrics
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import datetime
import json
import gzip
import os
import csv
import io
import re
import string
import glob
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt     

os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

file_path = glob.glob('Data\Sent_Analysis\Clean_Data\All_Cols\*.gz')

result_path = glob.glob('Data\Sent_Analysis\Results\orig_text\*.csv')

clean = pd.read_parquet(file_path, engine='pyarrow')

sum_results = []
text_results = []

    
for filename in result_path:
    df = pd.read_csv(filename, index_col=None, header=0)
    text_results.append(df)

results_frame = pd.concat(text_results, axis=0, ignore_index=True)




summary_frame = sum_frame.drop('Unnamed: 0', axis = 1)
text_frame = results_frame.drop('Unnamed: 0', axis = 1)

almost_final = pd.merge(summary_frame, text_frame, how = 'inner', on = ['idx'])

final = pd.merge(clean, text_frame, how = 'right', on = ['idx'])

final_head = final.head()

for col in final_head.columns:
    print(col)
    
final_final = final[['title', 'overall', 'reviewText', 'summary', 'original_text Compound Sentiment Score']]



text_conditions = [
    (final['original_text Compound Sentiment Score'] > 0.5),
    (final['original_text Compound Sentiment Score'] < -0.5),
    (final['original_text Compound Sentiment Score'] <= 0.5) & (final['original_text Compound Sentiment Score'] >= -0.5)
]

orig_conditions = [
    (final['overall'] == 5) | (final['overall'] == 4),
    (final['overall'] == 2) | (final['overall'] == 1),
    (final['overall'] == 3)
    
]


summary_conditions = [
    (final['original_summary Compound Sentiment Score'] > 0.5),
    (final['original_summary Compound Sentiment Score'] < -0.5),
    (final['original_summary Compound Sentiment Score'] <= 0.5) & (final['original_summary Compound Sentiment Score'] >= -0.5)
]

values = ['Positive', 'Negitive', 'Neutral']

final_final['text_sentiment'] = np.select(text_conditions, values)   
final_final['orig_sentiment'] = np.select(orig_conditions, values)  

x = final_final.head(100)

final_text_conditions = [
    (final_final['text_sentiment'] == 'Positive'),
    (final_final['text_sentiment'] == 'Negative'),
    (final_final['text_sentiment'] == 'Neutral')
]

final_orig_conditions = [
    (final_final['orig_sentiment'] == 'Positive'),
    (final_final['orig_sentiment'] == 'Negative'),
    (final_final['orig_sentiment'] == 'Neutral')
]

final_values = [2, 0, 1]

final_final['orig_sentiment'].value_counts()
final_final['text_sentiment'].value_counts()


final_final['text_score'] = np.select(final_text_conditions, final_values)   
final_final['orig_score'] = np.select(final_orig_conditions, final_values)

metrics.f1_score(final_final['orig_score'], final_final['text_score'], average='weighted') 

metrics.precision_recall_fscore_support(final_final['orig_score'], final_final['text_score'], average='weighted') 

cm = metrics.confusion_matrix(final_final['orig_score'], final_final['text_score']) 

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(values); ax.yaxis.set_ticklabels(values);
 



# Plot number of reviews by subcategory and rating
sum_pivot = pd.pivot_table(final_final, values = 'title', index = 'summary_sentiment', columns = 'overall', aggfunc= len)
sum_pivot = sum_pivot.reset_index()
sum_pivot.plot.bar(x = 'summary_sentiment', y = [1,2,3,4,5], rot = 50)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Summary Reviews by Sentiment")
plt.legend(title="Rating", loc='upper left', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:1.1f}M'.format(y*1e-6) for y in current_values])
plt.show()

# Plot number of reviews by subcategory and rating
text_pivot = pd.pivot_table(final_final, values = 'title', index = 'overall', columns = 'text_sentiment', aggfunc= 'count')
text_pivot = text_pivot.reset_index()
text_pivot.plot.bar(x = 'overall', y = ['Negitive', 'Neutral', 'Positive'], rot = 50)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Text Reviews by Sentiment")
plt.legend(title="Rating", loc='upper left', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:1.1f}M'.format(y*1e-6) for y in current_values])
plt.show()







x = final.drop_duplicates()

x = final.head(100)

y = clean.head(100)

z = frame.head(100)

a = x.head(100)


conditions = [
    (final['original_text Compound Sentiment Score'] > 0.5),
    (final['original_text Compound Sentiment Score'] < -0.5),
    (final['original_text Compound Sentiment Score'] <= 0.5) & (final['original_text Compound Sentiment Score'] >= -0.5)
]

values = ['Positive', 'Negitive', 'Neutral']

final['Sentiment'] = np.select(conditions, values)

final['Sentiment'].unique()


# Plot number of reviews by subcategory and rating
pivot = pd.pivot_table(final, values = 'idx', index = 'Sentiment', columns = 'overall', aggfunc='count')
pivot = pivot.reset_index()
pivot.plot.bar(x = 'sub_category', y = [1,2,3,4,5], rot = 50)
plt.xlabel("Sub Category")
plt.ylabel("Number of Reviews")
plt.title("Reviews by Sub Category and Rating (millions)")
plt.legend(title="Rating", loc='best', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:1.0f}M'.format(y*1e-6) for y in current_values])
plt.show()







print(df)

x = pd.read_csv('./Data/chunked\chunk_0.csv')
y = pd.read_csv('./Data/chunked\chunk_0.csv')
z = pd.read_csv('./Data/chunked\chunk_0.csv')



clean.loc[clean['idx'] == 14575]
