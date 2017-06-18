import re

from features.word_based.spam_regexes import allRegexes
from model import Feature

class SpamWords(Feature):
    FEATS = ["has_spam_words", "number_of_spam_words"]

    def transform(self, dataset):
        for instance in dataset:
            spamWords = 0
            for regex in allRegexes:
                spamWords += len(re.findall(regex, instance["content"]))
            instance["features"]["has_spam_words"] = 1 if spamWords > 0 else 0
            instance["features"]["number_of_spam_words"] = spamWords


        return dataset
