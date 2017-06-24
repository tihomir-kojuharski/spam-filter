import re

from model import Feature

class NumberOfWords(Feature):
    FEATS = ["number_of_words"]

    def transform(self, dataset):
        for instance in dataset:
            instance["content_words_str"] = re.sub(rb"[^\w\s\b]", b"", instance["content"])
            instance["content_words_list"] = instance["content_words_str"].split()
            instance["features"]["number_of_words"] = len(instance["content_words_list"])

        return dataset