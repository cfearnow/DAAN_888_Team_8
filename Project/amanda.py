# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from sklearn import metrics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set directories

# Change working directory to parent directory (where data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

# Identify paths for data ingest
# Original dataset path
file_path = glob.glob('Data\Sent_Analysis\Clean_Data\All_Cols\*.gz')

# Aanlysis data path
data_path = glob.glob('Data\Amanda_Sample/*.csv')

# Path where results will be read from
summary_result_path = 'Data\Sent_Analysis\Results\Amanda_Analysis\Amanda_Analysis_original_summary.csv'
text_result_path = 'Data\Sent_Analysis\Results\Amanda_Analysis\Amanda_Analysis_original_text.csv'

# read original data
clean = pd.read_parquet(file_path, engine='pyarrow')
#%%

# Set directories for analysis
#####    UPDATE BEFORE RUNNING    ###########
results_folder = 'Data/Sent_Analysis/Results/Amanda_Analysis/'
results_name = 'orig_summary_sent_'
col_names = ['idx', 'original_summary']
text_field = 'original_summary'
#############################################

# Identify the analyzer for use in VADER analysis
sid_analyzer = SentimentIntensityAnalyzer()

# Write functions for sentiment analysis
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
#%%
# Create fintion for reading analyzing and writing analysis results
def sentiment(file_path, sentiment_column):
            
    folder = results_folder
    return_name = 'Amanda_Analysis'
    
    folder_check = os.path.isdir(folder)
    
    if not folder_check:
        os.makedirs(folder)
        print("created folder : ", folder)
    else:
        print(folder, "already exists.")
        
    for file in file_path:
        name = return_name + '_' + text_field
        
        file_name = (results_folder + name +'.csv')
        file_check = os.path.isdir(file_name)
        
        if not file_check:
            print('Reading: ' + file)
            x = pd.read_csv(file, usecols = col_names)
            print('Analyzing: ' + file)
            y = get_sentiment_scores(x, sentiment_column)
            print('Writing: ' + name)
            y.to_csv(file_name, encoding='utf-8')
            print("created file : ", name)
        else:
            print(name, " already exists.")
            
# Run function
sentiment(data_path, text_field)             
#%% Analyze results

# Create two emply lists for analysis results (summary and text) 
sum_results = pd.read_csv(summary_result_path, index_col=None, header=0)
text_results = pd.read_csv(text_result_path, index_col=None, header=0)
    

summary_results_frame = sum_results.drop(['Unnamed: 0', 'original_summary'], axis = 1)
text_results_frame = text_results.drop(['Unnamed: 0', 'original_text'], axis = 1)

data_merge = pd.merge(summary_results_frame, text_results_frame, how = 'inner', on = ['idx'])

final_merge = pd.merge(clean, data_merge, how = 'right', on = ['idx'])

final_head = final_merge.head()

for col in final_head.columns:
    print(col)
    
final = final_merge[['idx', 'title', 'overall', 'reviewText', 'summary', 'original_text Compound Sentiment Score', 'original_summary Compound Sentiment Score']]

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

final['text_sentiment'] = np.select(text_conditions, values)   
final['orig_sentiment'] = np.select(orig_conditions, values) 
final['summary_sentiment'] = np.select(summary_conditions, values) 

x = final.head(100)

final_text_conditions = [
    (final['text_sentiment'] == 'Positive'),
    (final['text_sentiment'] == 'Negative'),
    (final['text_sentiment'] == 'Neutral')
]

final_orig_conditions = [
    (final['orig_sentiment'] == 'Positive'),
    (final['orig_sentiment'] == 'Negative'),
    (final['orig_sentiment'] == 'Neutral')
]

final_summ_conditions = [
    (final['summary_sentiment'] == 'Positive'),
    (final['summary_sentiment'] == 'Negative'),
    (final['summary_sentiment'] == 'Neutral')
]

final_values = [2, 0, 1]

final['orig_sentiment'].value_counts()
final['text_sentiment'].value_counts()
final['summary_sentiment'].value_counts()


final['text_score'] = np.select(final_text_conditions, final_values)   
final['orig_score'] = np.select(final_orig_conditions, final_values)
final['summary_score'] = np.select(final_summ_conditions, final_values)

x = final.head(100)

metrics.f1_score(final['orig_score'], final['text_score'], average='weighted') 

metrics.precision_recall_fscore_support(final['orig_score'], final['text_score'], average='weighted') 

cm = metrics.confusion_matrix(final['orig_score'], final['text_score']) 

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(values); ax.yaxis.set_ticklabels(values);
 

# Plot number of reviews by subcategory and rating
text_pivot = pd.pivot_table(final, values = 'title', index = 'overall', columns = 'text_sentiment', aggfunc= 'count')
text_pivot = text_pivot.reset_index()
text_pivot.plot.bar(x = 'overall', y = ['Negitive', 'Neutral', 'Positive'], rot = 50)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Text Reviews by Overall Star Rating and Sentiment")
plt.legend(title="Rating", loc='upper left', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
plt.show()

# Plot number of reviews by subcategory and rating
sum_pivot = pd.pivot_table(final, values = 'title', index = 'overall', columns = 'summary_sentiment', aggfunc= 'count')
sum_pivot = sum_pivot.reset_index()
sum_pivot.plot.bar(x = 'overall', y = ['Negitive', 'Neutral', 'Positive'], rot = 50)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Summary Reviews by Overall Star Rating and Sentiment")
plt.legend(title="Rating", loc='upper left', fontsize='small', fancybox=True)
current_values = plt.gca().get_yticks()
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





