from model import Feature
import re


class DigitRatio(Feature):
    FEATS = ["digit_ratio"]

    def transform(self, dataset):
        for instance in dataset:
            instance["features"]["digit_ratio"] = len(re.findall(b"\d", instance["content"]))     \
                                                  / instance["features"]["number_of_characters"]

        return dataset
