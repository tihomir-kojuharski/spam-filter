from model import Feature

class NumberOfCharacters(Feature):
    FEATS = ["number_of_characters"]

    def transform(self, dataset):
        for instance in dataset:
            instance["features"]["number_of_characters"] = len(instance["content"])

        return dataset
