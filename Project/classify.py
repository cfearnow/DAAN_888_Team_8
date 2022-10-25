import os
import glob
from pynvml import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline

import torch

torch.cuda.is_available()

# Change working directory
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

# Directory to chunked results
df = pd.read_csv('final_data_set.csv')

# Directory to chunked results
chunked_path = './Data/classify/chunked'

classify_path = './Data/Sent_Analysis/Classify_Results'

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
            #varname['tokenized_sents'] = varname.apply(lambda row: nltk.word_tokenize(row['reviewText']), axis=1)
            varname.to_csv(folder_path, encoding='utf-8')
            print("created file : ", new_name)
        else:
            print(new_name, " already exists.")

chunk_data(df)

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    
def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
    
print_gpu_utilization()

# Identify the model we will use
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", 
                      device = 0)

candidate_labels = ['package',
                    'price',
                    'expected',
                    'pleased',
                    'clean',
                    'quality',
                    'invoice',
                    'customer service',
                    'damage',
                    'speed',
                    'bargain',
                    'complain',
                    'contact',
                    'shipping',
                    'help',
                    'difficult',
                    'disclose',
                    'offensive',
                    'material',
                    'advertise',
                    'rating',
                    'size',
                    'workmanship',
                    'issue'
                    ]
    
 #%%

file_chunks = glob.glob(chunked_path + '/*.csv')

#####    UPDATE BEFORE RUNNING    ###########
 
classify_results_folder = 'Data/Sent_Analysis/Classify_Results/'
results_name = 'classify_'
col_names = ['idx', 'reviewText']
text_field = 'reviewText'

#############################################   
    
def classify(file_path):
            
    folder = classify_results_folder
    
    folder_check = os.path.isdir(folder)
    
    if not folder_check:
        os.makedirs(folder)
        print("created folder : ", folder)
    else:
        print(folder, "already exists.")
        
    for file in file_path:
        emp_str = ""
        for m in file:
            if m.isdigit():
                emp_str = emp_str + m
        name = results_name + emp_str
        
        file_name = (classify_results_folder + name +'.csv')
        file_check = os.path.isdir(file_name)
        
        if not file_check:
            print('Reading: ' + file)
            x = pd.read_csv(file, usecols = col_names)
            print('Analyzing: ' + file)
            predictedCategories = []
            for i in tqdm(range(len(x))):
                text = x.iloc[i,]['reviewText']
                #idx_num = x.iloc[i,]['idx']
                res = classifier(text, candidate_labels, multi_label=True)#setting multi-class as True
                labels = res['labels'] 
                scores = res['scores'] #extracting the scores associated with the labels
                res_dict = {label : score for label,score in zip(labels, scores)}
                sorted_dict = dict(sorted(res_dict.items(), key=lambda x:x[1],reverse = True)) #sorting the dictionary of labels in descending order based on their score
                categories  = []
                for i, (k,v) in enumerate(sorted_dict.items()):
                    if(i > 4): #storing only the best 5 predictions
                        break
                    else:
                        categories.append([k, v])
                predictedCategories.append(categories)
                pred_df = pd.DataFrame(predictedCategories)  
            print('Writing: ' + name)
            pred_df.to_csv(file_name, encoding='utf-8')
            print("created file : ", name)
        else:
            print(name, " already exists.")
            
classify(file_chunks)

#%%

classify_chunks = glob.glob(classify_path + '/*.csv')

merge_dict = {}
for key in file_chunks:
    for value in classify_chunks:
        merge_dict[key] = value
        classify_chunks.remove(value)
        break

#####    UPDATE BEFORE RUNNING    ###########
 
merged_results_folder = 'Data/Merged/Classify_Orig_Results/'
merged_results_name = 'classify_orig_'
comma_column = ['0','1','2','3','4']

#############################################   
    
def merged(dictionary):
            
    folder = merged_results_folder
    
    folder_check = os.path.isdir(folder)
    
    if not folder_check:
        os.makedirs(folder)
        print("created folder : ", folder)
    else:
        print(folder, "already exists.")
        
    for k, v in merge_dict.items():
        emp_str = ""
        for m in k:
            if m.isdigit():
                emp_str = emp_str + m
        name = merged_results_name + emp_str
        file_name = (merged_results_folder + name +'.csv')
        file_check = os.path.isdir(file_name)
        
        if not file_check:
            print('Reading: ' + k)
            print('Reading: ' + v)
            x = pd.read_csv(k)
            y = pd.read_csv(v)
            print('Merging Files')
            mergedDf = x.merge(y, left_index=True, right_index=True)
            print('Writing: ' + name)
            rid = mergedDf.drop(['Unnamed: 0.1', 'Unnamed: 0_x', 'Unnamed: 0_y'], axis = 1)
            
            for col in comma_column:
                rid[col] = rid[col].apply(lambda x: x.replace(" ", ""))
                rid[col] = rid[col].apply(lambda x: x.replace("'", ""))
                rid[col] = rid[col].apply(lambda x: x.replace("[", ""))
                rid[col] = rid[col].apply(lambda x: x.replace("[", ""))
                rid[['Class ' + col, 'Rating ' + col ]] = rid[col].str.split(',', expand=True) 
                
            rid = rid.drop(comma_column, axis = 1)
            rid.to_csv(file_name, encoding='utf-8')
            print("created file : ", name)
        else:
            print(name, " already exists.")
            
merged(merge_dict)

# Read and append all chunked data

final_files = glob.glob(os.path.join(merged_results_folder, "*.csv"))

final_df = pd.concat((pd.read_csv(f) for f in final_files), ignore_index=True)
final_df = final_df.drop(['Unnamed: 0'], axis = 1)

x = final_df.head()

final_df.to_csv('classified_file.csv', encoding='utf-8')

final_df['Class 0'].value_counts().plot(kind = 'bar')

# Plot number of reviews by subcategory and rating
text_pivot = pd.pivot_table(final_df, values = 'title', index = 'Class 0', aggfunc= 'count')
text_pivot = text_pivot.reset_index()
text_pivot.plot.bar(x = 'Class 0', y = 'title', rot = 90)
plt.xlabel("Category")
plt.ylabel("Total Count")
plt.title("Classification by Category")
current_values = plt.gca().get_yticks()
plt.show()
