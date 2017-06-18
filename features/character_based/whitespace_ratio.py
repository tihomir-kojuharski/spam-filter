from model import Feature
import re


class WhitespaceRatio(Feature):
    FEATS = ["whitespace_ratio"]

    def transform(self, dataset):
        for instance in dataset:
            instance["features"]["whitespace_ratio"] = len(re.findall(b"\s", instance["content"]))     \
                                                  / instance["features"]["number_of_characters"]


        return dataset
