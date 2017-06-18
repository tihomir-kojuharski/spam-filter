# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:00:44 2017

@author: Abhijeet Singh
"""

import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

mailsCount = 0


class WordDesc:
    def __init__(self, wordId, counter):
        self.wordId = wordId
        self.counter = counter


def make_Dictionary(root_dir):
    emails_dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    all_words = []
    for emails_dir in emails_dirs:
        if not os.path.isdir(emails_dir): continue
        dirs = [os.path.join(emails_dir, f) for f in os.listdir(emails_dir)]

        for d in dirs:
            if not os.path.isdir(d): continue
            emails = [os.path.join(d, f) for f in os.listdir(d)]
            for mail in emails:
                if '.DS_Store' in mail: continue

                globals()['mailsCount'] = globals()['mailsCount'] + 1

                with open(mail, "rb") as m:
                    for line in m:
                        words = line.split()
                        all_words += words

    wordCountersList = Counter(all_words)

    items_to_remove = []
    for item in wordCountersList.keys():
        if item.isalpha() == False or len(item) == 1:
            items_to_remove.append(wordCountersList[item])
    for item in items_to_remove:
        del wordCountersList[item]

    wordCountersList = wordCountersList.most_common(3000)

    dictionary = {}
    nextWordId = 0

    for wordCounter in wordCountersList:
        dictionary[wordCounter[0]] = WordDesc(nextWordId, wordCounter[1])
        nextWordId += 1

    np.save('dict_enron.npy', wordCountersList)

    return dictionary


def extract_features(root_dir):
    emails_dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    docID = 0
    features_matrix = np.zeros((mailsCount, 3000))
    train_labels = np.zeros(mailsCount)
    for emails_dir in emails_dirs:
        if not os.path.isdir(emails_dir): continue
        dirs = [os.path.join(emails_dir, f) for f in os.listdir(emails_dir)]
        for d in dirs:
            if not os.path.isdir(d):
                continue
            emails = [os.path.join(d, f) for f in os.listdir(d)]
            for mail in emails:
                if '.DS_Store' in mail: continue
                print(mail)
                with open(mail, 'rb') as m:
                    all_words = []
                    for line in m:
                        words = line.split()
                        all_words += words

                    wordCountersList = Counter(all_words)

                    for word, wordCount in wordCountersList.items():
                        wordDesc = dictionary.get(word)
                        if wordDesc == None:
                            continue
                        features_matrix[docID, wordDesc.wordId] = wordCount
                train_labels[docID] = int(mail.split(".")[-2] == 'spam')
                docID = docID + 1
    return features_matrix, train_labels


# Create a dictionary of words with its frequency

root_dir = 'enron-dataset'
dictionary = make_Dictionary(root_dir)
# dictinary = np.load('dict_enron.npy')

# Prepare feature vectors per training mail and its labels

features_matrix, labels = extract_features(root_dir)
np.save('enron_features_matrix.npy', features_matrix)
np.save('enron_labels.npy', labels)

# features_matrix = np.load('enron_features_matrix.npy');
# labels = np.load('enron_labels.npy');

print(features_matrix.shape)
print(labels.shape)
print(sum(labels == 0), sum(labels == 1))
X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)

## Training models and its variants

classifiers = [
    LinearSVC(),
    MultinomialNB(),
    KNeighborsClassifier(1)
]

for model in classifiers:
    model.fit(X_train, y_train)
    result = model.predict(X_test)
    print("==================\n{0}:\n{1}".format(str(model.__class__.__name__), confusion_matrix(y_test, result)))
