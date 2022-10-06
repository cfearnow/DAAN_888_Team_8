import os
import glob
from pynvml import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline

# Change working directory
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')
folder_path = 'Data\Checking'

# Read file for test
test_file = pd.read_csv(folder_path + '\Toys_Games.csv')

test_file2 = test_file[test_file['reviewText'].notnull()]

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
classifier = pipeline("zero-shot-classification", device = 0)


candidate_labels = [#'ship',
                    #'shipped',
                    'shipping',
                    'package',
                    #'damage',
                    'damaged',
                    #'delay',
                    'delayed',
                    'missing',
                    'ups',
                    'fedex',
                    'delivery',
                    'fast',
                    'slow',
                    'shattered',
                    'mangled',
                    'crushed',
                    'burst',
                    'shredded',
                    'freight',
                    'supply',
                    'usps',
                    'mail',
                    'lasership',
                    'dhl',
                    'logistics',
                    'expensive',
                    'costly',
                    'credit',
                    'money',
                    'dollar',
                    #'dollars',
                    'inexpensive', 
                    'bargain',
                    'rate',
                    #'charge',
                    #'charged',
                    'charges',
                    'fee',
                    #'fees',
                    'deduction',
                    'discount',
                    'overcharge',
                    'surcharge',
                    'tax',
                    'markdown',
                    'bill',
                    'invoice', 
                    'complaint',
                    'bad',
                    'excellent',
                    'manager',
                    'call',
                    #'called',
                    'customer',
                    'service',
                    'chat',
                    'support',
                    'professional',
                    'unprofessional',
                    'rude',
                    'helpful',
                    'unhelpful',
                    'yell',
                    'scam',
                    'friendly',
                    'email',
                    #'emailed',
                    'broken',
                    'missing',
                    'scratched',
                    'pristine',
                    #'matches',
                    'matched',
                    'expected',
                    #'bust',
                    'busted',
                    'snapped',
                    'ripped',
                    'torn',
                    'chipped',
                    'faulty',
                    'expired',
                    'spoiled',
                    #'flaw',
                    'flawed',
                    'flawless',
                    'blemished']

candidate_labels = ['shipping',
                    'package',
                    'speed',
                    'charged',
                    'cost',
                    'bargain',
                    'service',
                    'manager',
                    'unhelpful',
                    'complain',
                    'call',
                    'expected',
                    'damage',
                    'clean',
                    'pleased',
                    'quality']

predictedCategories = []
for i in tqdm(range(len(test_file))):
    text = test_file2.iloc[i,]['reviewText']
    number = test_file2.iloc[i,]['idx']
    #cat = test_file.iloc[i,]['categories']
    #cat = cat.split()
    res = classifier(text, candidate_labels, multi_label=True)#setting multi-class as True
    labels = res['labels'] 
    scores = res['scores'] #extracting the scores associated with the labels
    res_dict = {label : score for label,score in zip(labels, scores)}
    sorted_dict = dict(sorted(res_dict.items(), key=lambda x:x[1],reverse = True)) #sorting the dictionary of labels in descending order based on their score
    categories  = []
    for i, (k,v) in enumerate(sorted_dict.items()):
        if(i > 2): #storing only the best 3 predictions
            break
        else:
            categories.append([k, v])
    predictedCategories.append(categories)


fourth_run = pd.DataFrame(predictedCategories)
fourth_run.to_csv('fourth_run.csv', encoding='utf-8')
test_file2.to_csv('test_data.csv', encoding='utf-8')
