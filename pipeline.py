from csv import DictReader

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# from features.counting_feat import SentenceLength, BagOfTfIDF, WordOverlap
# from features.nltk_feat import POS, NER
# from features.preprocess import TokenizedLemmas
from features.alpha_ratio import AlphaRatio
from features.digit_ratio import DigitRatio
from features.number_of_characters import NumberOfCharacters
from features.special_chars_ratio import SpecialCharsRatio
from features.whitespace_ratio import WhitespaceRatio
from features.word_counts import WordCounts

from model import ToMatrix

import os

dataset_dir = "enron-dataset"


def get_dataset():
    emails_dirs = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]

    instances = []
    for emails_dir in emails_dirs:
        if not os.path.isdir(emails_dir): continue
        dirs = [os.path.join(emails_dir, f) for f in os.listdir(emails_dir)]

        for d in dirs:
            if not os.path.isdir(d): continue
            emails = [os.path.join(d, f) for f in os.listdir(d)]
            for mail in emails:
                if '.DS_Store' in mail: continue

                with open(mail, "rb") as mailDescriptor:
                    instances.append({
                        "content": mailDescriptor.read(),
                        "is_spam": int(mail.split(".")[-2] == 'spam'),
                        "features": {}
                    })

    return instances


def get_features(train):
    features = [
        ('word_counts', WordCounts(train)),
        ('number_of_characters', NumberOfCharacters()),
        ('alpha_ratio', AlphaRatio()),
        ('digit_ratio', DigitRatio()),
        ('whitespace_ratio', WhitespaceRatio()),
        ('special_chars_ratio', SpecialCharsRatio())
    ]
    return features


def get_pipeline(features):
    feature_names = []
    for feature in features:
        feature_names += feature[1].FEATS
    print(feature_names)
    return Pipeline(features + [('transform', ToMatrix(features=feature_names)),
                                ('norm', MinMaxScaler())])


def run_classifiers(test, train):
    classifiers = [
        SVC(kernel="linear", C=0.025),
        MultinomialNB(),
        KNeighborsClassifier(1)
    ]

    print("Transforming data to features...")
    #  get pipeline of features and generate them
    pipeline = get_pipeline(get_features(train))
    X_train = pipeline.fit_transform(train)
    trainLabels = [instance['is_spam'] for instance in train]

    testLabels = [instance["is_spam"] for instance in test]
    # get the features for the test set
    X_test = pipeline.fit_transform(test)

    print("Finished data to features...")

    for classifier in classifiers:
        print("==================== {0} ==================== ".format(str(classifier.__class__.__name__)))

        print("Start training...")
        classifier.fit(X_train, trainLabels)
        print("Finished training...")

        print("Starting prediction...")
        predictedLabels = classifier.predict(X_test)
        print("Finished prediction...")

        print("\nConfusion matrix:\n{0}".format(confusion_matrix(testLabels, predictedLabels)))


if __name__ == "__main__":
    train, test = train_test_split(get_dataset(), test_size=0.40)
    predictedLabels = run_classifiers(test, train)
