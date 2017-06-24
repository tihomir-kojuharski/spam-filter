from model import Feature
import re

class ShortWordsRatio(Feature):
    FEATS = ["short_words_ratio"]

    def transform(self, dataset):
        for instance in dataset:
            instance["features"]["short_words_ratio"] = len(re.findall(b"\b\w{,2}\b", instance["content_words_str"])) \
                                                        / instance["features"]["number_of_words"]

        return dataset
