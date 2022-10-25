import os
import glob
from pynvml import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from transformers import pipeline

import torch

torch.cuda.is_available()

# Change working directory
os.chdir('D:\School_Files\DAAN_888\Team_8_Project')

# Directory to chunked results
df = pd.read_csv('final_data_set.csv')

df['new_text'] = df['reviewText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

df['new_text2'] = df['new_text'].apply(wordnet_lemmatizer.lemmatize)

df['is_equal'] = (df['new_text2'] == df['new_text'])

df.to_csv('recheck.csv', encoding='utf-8')






























# Directory to chunked results
chunked_path = './Data/classify/chunked'

classify_path = './Data/Sent_Analysis/Classify_Results'