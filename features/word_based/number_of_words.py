from model import Feature

class NumberOfWords(Feature):
    FEATS = ["number_of_words"]

    def transform(self, dataset):
        for instance in dataset:
            instance["content_words"] = instance["content"].split()
            instance["features"]["number_of_words"] = len(instance["content_words"])

        return dataset