import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Importing the dataset
data_set = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Cleaning the dataset
# drop unnecessary columns
data_set = data_set.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# rename columns to something better
data_set.columns = ['label', 'message']

# create binary labels
data_set['b_label'] = data_set['label'].map({'ham': 0, 'spam': 1})

X = data_set.iloc[:,1].values
y = data_set.iloc[:,-1].values

# print(X)

# Tokenized
# tfifd_vectorizer = TfidfVectorizer(decode_error='ignore')
# X = tfifd_vectorizer.fit_transform(X).toarray()

count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(X).toarray()

# rows, cols = np.nonzero(X)

# np.set_printoptions(threshold=np.nan)

# print(X[rows, cols])


# Splitting dataset into Training and Test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20)

model = MultinomialNB()
model.fit(X_train, Y_train)

y_predict = model.predict(X_test)

print("train score:", model.score(X_train, Y_train))
print("test score:", model.score(X_test, Y_test))

def visualize_data(label):
    words = ''
    for msg in data_set[data_set['label'] == label]['message']:
        words += msg.lower() + ' '
    word_cloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()

# visualize_data('spam')
# visualize_data('ham')