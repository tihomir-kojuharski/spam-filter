from model import Feature
import re

class AverageWordLen(Feature):
    FEATS = ["average_word_len"]

    def transform(self, dataset):
        for instance in dataset:
            instance["features"]["average_word_len"] = len(re.findall(b"\w", instance["content_words_str"])) \
                                                        / instance["features"]["number_of_words"]

        return dataset
