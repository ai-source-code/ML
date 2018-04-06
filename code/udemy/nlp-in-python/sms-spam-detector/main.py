import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Importing the dataset
data_set = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Cleaning the dataset
# drop unnecessary columns
data_set = data_set.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# rename columns to something better
data_set.columns = ['label', 'message']

# create binary labels
data_set['b_label'] = data_set['label'].map({'ham': 0, 'spam': 1})

X = data_set.iloc[:,1]
y = data_set.iloc[:,-1]

def visualize_data(label):
    words = ''
    for msg in data_set[data_set['label'] == label]['message']:
        words += msg.lower() + ' '
    word_cloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()

visualize_data('spam')