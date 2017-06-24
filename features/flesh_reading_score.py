from textstat.textstat import textstat
from model import Feature

class FleschReadingEase(Feature):
    FEATS = ["flesch_reading_score"]

    def transform(self, dataset):
        for instance in dataset:
            try:
                content = instance["content"].decode('ascii')
                instance["features"]["flesch_reading_score"] = textstat.flesch_reading_ease(content)
            except Exception as e:
                instance["features"]["flesch_reading_score"] = 0

        return dataset
