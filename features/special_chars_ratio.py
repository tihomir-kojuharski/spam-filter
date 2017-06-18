from model import Feature
import re


class SpecialCharsRatio(Feature):
    FEATS = ["special_chars_ratio"]

    def transform(self, dataset):
        for instance in dataset:
            contentLen = len(instance["content"])
            instance["features"]["special_chars_ratio"] = len(re.findall(b"[^\w\s\d]", instance["content"]))     \
                                                  / instance["features"]["number_of_characters"]


        return dataset
