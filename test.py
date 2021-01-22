import pickle
import pandas as pd
import re
import nltk
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.metrics import classification_report
import sys

nltk.download('stopwords')
nltk.download('punkt')

tfidf = pickle.load(open('tfidf.pickle', 'rb'))
clf = pickle.load(open('lr_clf_res.pickle', 'rb'))
stemmer = PorterStemmer()
stop_words = stopwords.words('english')


def preprocess_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub('\s\s+', ' ', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    text = [stemmer.stem(word) for word in word_tokenize(text) if word not in stop_words]
    return text


def prepare_data(data):
    data['reviewText'] = data['reviewText'].astype(str)
    data['summary'] = data['summary'].astype(str)
    data['review_and_summary_preprocessed'] = data['reviewText'] + ' ' + data['summary']
    data['review_and_summary_preprocessed'] = data['review_and_summary_preprocessed'].apply(preprocess_text)
    list_of_docs_test = [' '.join(words) for words in data['review_and_summary_preprocessed'].tolist()]
    X_test_tfidf = tfidf.transform(list_of_docs_test)
    return X_test_tfidf


data = pd.read_csv(sys.argv[1])
X = prepare_data(data)
y = data['score'].astype(int).tolist()
y_pred = clf.predict(X)
print(classification_report(y, y_pred))
