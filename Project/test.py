# Python imports
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import FAST_VERSION
print(f"gensim FAST_VERSION = {FAST_VERSION}. <-- Hopefully that says '1' not '-1'")

from sklearn.manifold import TSNE   # actually TNSE is the speed bottleneck, not Word2Vec
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm   
import nltk
import re
import codecs
import multiprocessing
import os
import sys
import matplotlib.patheffects as PathEffects
import tempfile
import imageio
import shutil

# Change working directory to parent directory (where data is stored)
os.chdir('D:\School_Files\DAAN_888\Team_8_Project\google_news')

#  plot_type: 'notebook' allows for interactive plots (=better!), but Colab 
#       only supports 'inline'.
#       For interactive plots, best to execute one cell at a time (manually), rather 
#       than Kernal > Run All, because interactives will appear blank until all 
#       code cells have executed (whereas inline plots render immediately).
plot_type = 'inline' if 'google.colab' in sys.modules else 'notebook' # Auto-detect Colab
%matplotlib $plot_type

model_gn = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
