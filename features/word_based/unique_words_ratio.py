from model import Feature
import re

class UniqueWordsRatio(Feature):
    FEATS = ["unique_words_ratio"]

    def transform(self, dataset):
        for instance in dataset:
            uniqueWords = {}
            for word in instance["content_words_list"]:
                uniqueWords[word] = True

            instance["features"]["unique_words_ratio"] = len(uniqueWords.keys()) \
                                                        / instance["features"]["number_of_words"]

        return dataset
