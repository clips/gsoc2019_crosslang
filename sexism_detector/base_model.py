import pandas as pd
import re

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


# Raw dataset (with original preprocessing)



# No stopwords, no punctuation, no regex

sexism_no_stopwords_test = pd.read_csv('TemporalCorpora\\test_no_stop_and_regex.csv')
sexism_no_stopwords_train = pd.read_csv('TemporalCorpora\\train_no_stop_and_regex.csv')
sexism_no_stopwords_test['Label'] = sexism_no_stopwords_test['Label'].str.replace(';', '')
sexism_no_stopwords_train['Label'] = sexism_no_stopwords_train['Label'].str.replace(';', '')
remove_said_regexp = re.compile('.*писал\(а\):')
sexism_no_stopwords_train['Text'] = sexism_no_stopwords_train['Text'].str.replace(remove_said_regexp, '')
sexism_no_stopwords_test['Text'] = sexism_no_stopwords_test['Text'].str.replace(remove_said_regexp, '')


# Lemmatized version


# Sklearn version of separation to test and train
# Just to be on a safe side
sexism_test = pd.read_csv('TemporalCorpora\\test.csv')
sexism_train = pd.read_csv('TemporalCorpora\\train.csv')
sexism_full = pd.concat([sexism_test, sexism_train], ignore_index=True)
sexism_full['Label'] = sexism_full['Label'].str.replace(';', '')
remove_said_regexp = re.compile('.*писал\(а\):')
remove_n_and_r = re.compile('[\r\n]')
sexism_full['Text'] = sexism_full['Text'].str.replace(remove_said_regexp, '')
sexism_full['Text'] = sexism_full['Text'].str.replace(remove_n_and_r, '')


# Visual representation

plt.style.use('ggplot')
labels_count = sexism_full['Label'].value_counts()
plt.figure(figsize=(10,5))
plt.pie(labels_count.values, explode=[0, 0.1], labels = labels_count.index)
plt.title('Current data-set by labels')
#plt.show()


# Labels to integer representation

labels_enc = LabelEncoder()
y_full = labels_enc.fit_transform(sexism_full['Label'])



tfidf_vec = TfidfVectorizer(min_df=5)



