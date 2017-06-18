from collections import Counter
import numpy as np
from model import Feature

class WordDesc:
    def __init__(self, wordId, counter):
        self.wordId = wordId
        self.counter = counter

class WordCounts(Feature):
    FEATS = ["words_counts"]
    dictionary = {}

    def __init__(self, train_data):
        all_words = []

        for instance in train_data:
            for line in instance["content"].split(b"\n"):
                all_words += line.rstrip().split()

        wordCountersList = Counter(all_words)

        items_to_remove = []
        for item in wordCountersList.keys():
            if item.isalpha() == False or len(item) == 1:
                items_to_remove.append(wordCountersList[item])
        for item in items_to_remove:
            del wordCountersList[item]

        wordCountersList = wordCountersList.most_common(3000)

        nextWordId = 0

        for wordCounter in wordCountersList:
            self.dictionary[wordCounter[0]] = WordDesc(nextWordId, wordCounter[1])
            nextWordId += 1

        np.save('dict_enron.npy', wordCountersList)

    def transform(self, dataset):
        for instance in dataset:
            all_words = []

            for line in instance["content"].split():
                all_words += line.split()

            wordCountersList = Counter(all_words)

            wordsCounts = [0] * 3000

            for word, wordCount in wordCountersList.items():
                wordDesc = self.dictionary.get(word)
                if wordDesc == None:
                    continue
                wordsCounts[wordDesc.wordId] = wordCount

            instance["features"]["words_counts"] = wordsCounts

        return dataset
