# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from collections import Counter


# Importing the dataset
data_set = pd.read_csv('Reviews.csv', delimiter = ',', nrows = 1000)

# Cleaning the data
corpus = []
data_set_review = data_set['Text']
for index in range(len(data_set_review)):
    review = data_set_review[index]
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    porter_stemmer = PorterStemmer()
    review = [porter_stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
   
    review = ' '.join(review)
    corpus.append(review)
    

word_counter = Counter(data_set_review.split())
print(word_counter)
# print(len(corpus))
