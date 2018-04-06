import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

data_set = pd.read_csv('train.csv', nrows = 100000)

data_set = data_set.drop(['id', 'qid1', 'qid2'], axis = 1)

X = data_set.iloc[:, 0:-1].values
y = data_set.iloc[:, -1].values

def visualize_data(X, y):
    question1_words = ''
    question2_words = ''
    for index in range(len(X)):
        question1_words += X[index][0].lower() + ''
        question2_words += X[index][1].lower() + ''
    word_cloud = WordCloud(width=600, height=400).generate(question1_words)
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.title('Question 1 word cloud')
    plt.show()
    word_cloud = WordCloud(width=600, height=400).generate(question2_words)
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.title('Question 2 word cloud')
    plt.show()

visualize_data(X, y)
