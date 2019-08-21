import pandas as pd
import re

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

labels_enc = LabelEncoder()
tfidf_vec = TfidfVectorizer(min_df=5)


# Just to be on a safe side, I make sklearn split and check with it too

sexism_test = pd.read_csv('TemporalCorpora\\test_no_stop_and_regex.csv')
sexism_train = pd.read_csv('TemporalCorpora\\train_no_stop_and_regex.csv')
sexism_full = pd.concat([sexism_test, sexism_train], ignore_index=True)
sexism_full['Label'] = sexism_full['Label'].str.replace(';', '')
remove_said_regexp = re.compile('.*писал\(а\):')
remove_n_and_r = re.compile('[\r\n]')
sexism_full['Text'] = sexism_full['Text'].str.replace(remove_said_regexp, '')
sexism_full['Text'] = sexism_full['Text'].str.replace(remove_n_and_r, '')
y_full = labels_enc.fit_transform(sexism_full['Label'])
X_full_tfidf = tfidf_vec.fit_transform(sexism_full['Text'])

X_full_tfidf = tfidf_vec.fit_transform(sexism_full['Text'])
x_tr,x_test,y_tr,y_test = train_test_split(X_full_tfidf, y_full,stratify=y_full, test_size=0.1, train_size=0.9, shuffle=True)

gauss_nb_tfidf = GaussianNB()

def calculate_for_base_model(x_train,x_test,y_train,y_test):
    # We have to use balanced accuracy score, because our corpus in unballanced
    gauss_nb_tfidf.fit(x_train.toarray(), y_train)
    preds = gauss_nb_tfidf.predict(x_test.toarray())
    print(balanced_accuracy_score(y_test, preds))

#calculate_for_base_model(x_tr,x_test,y_tr,y_test)

logisticRegr = LogisticRegression()
def calculate_log_regr(x_train,x_test,y_train,y_test):
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_test)
    print(balanced_accuracy_score(y_test, predictions))

#calculate_log_regr(x_tr,x_test,y_tr,y_test)
