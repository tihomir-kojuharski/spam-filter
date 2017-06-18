from model import Feature
import re


class AlphaRatio(Feature):
    FEATS = ["alpha_ratio"]

    def transform(self, dataset):
        for instance in dataset:
            instance["features"]["alpha_ratio"] = len(re.findall(b"\w", instance["content"]))     \
                                                  / instance["features"]["number_of_characters"]

        return dataset
