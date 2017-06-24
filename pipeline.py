import os
from itertools import cycle

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from twinsvm.TVSVM import TwinSVMClassifier
import datetime

from features.character_based.alpha_ratio import AlphaRatio
from features.character_based.digit_ratio import DigitRatio
from features.character_based.number_of_characters import NumberOfCharacters
from features.character_based.special_chars_ratio import SpecialCharsRatio
from features.character_based.whitespace_ratio import WhitespaceRatio
from features.word_based.average_word_len import AverageWordLen
from features.word_based.number_of_words import NumberOfWords
from features.word_based.short_words_ratio import ShortWordsRatio
# from features.word_based.spam_words import SpamWords
from features.word_based.spam_words import SpamWords
from features.word_based.unique_words_ratio import UniqueWordsRatio
from features.word_counts import WordCounts
from features.flesh_reading_score import  FleschReadingEase
from model import ToMatrix
from neural_network import NeuralNetwork

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

    # return instances[3000:4000]
    return instances


def get_features(train):
    features = [
        ('word_counts', WordCounts(train)),
        ('number_of_characters', NumberOfCharacters()),
        ('alpha_ratio', AlphaRatio()),
        ('digit_ratio', DigitRatio()),
        ('whitespace_ratio', WhitespaceRatio()),
        ('special_chars_ratio', SpecialCharsRatio()),
        ('number_of_words', NumberOfWords()),
        ('short_words_ratio', ShortWordsRatio()),
        ('average_word_len', AverageWordLen()),
        ('unique_words_ratio', UniqueWordsRatio()),
        ('spam_words', SpamWords()),
        ('flesch_reading_score', FleschReadingEase()),
    ]
    return features


def get_pipeline(features):
    feature_names = []
    for feature in features:
        feature_names += feature[1].FEATS
    print(feature_names)
    return Pipeline(features + [
        ('transform', ToMatrix(features=feature_names)),
        ('norm', MinMaxScaler())
    ])


def run_classifiers(test, train):
    nnClassifier = NeuralNetwork(activation='sigmoid')

    params3 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1, 'kernel_type': 0, 'kernel_param': 1, 'fuzzy': 0}

    classifiers = [
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # TwinSVMClassifier(**params3),
        MultinomialNB(),
        # KNeighborsClassifier(1),
        # nnClassifier,
        # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis(),
    ]

    print("Transforming data to features...")
    #  get pipeline of features and generate them
    pipeline = get_pipeline(get_features(train))

    # X_train = np.load('enron_train_features_matrix.npy')
    # trainLabels = np.load('enron_train_labels.npy')
    X_train = pipeline.fit_transform(train)
    trainLabels = [instance['is_spam'] for instance in train]
    np.save('enron_train_features_matrix.npy', X_train)
    np.save('enron_train_labels.npy', trainLabels)

    X_test = pipeline.fit_transform(test)
    testLabels = [instance["is_spam"] for instance in test]
    # X_test = np.load('enron_test_features_matrix.npy');
    # testLabels = np.load('enron_test_labels.npy');
    np.save('enron_test_features_matrix.npy', X_test)
    np.save('enron_test_labels.npy', testLabels)

    nnClassifier.X_test = np.array(X_test)
    nnClassifier.Y_test = np.array(testLabels)

    print("Finished transforming data to features...")

    for classifier in classifiers:
        print("\n==================== {0} ====================".format(str(classifier.__class__.__name__)))

        print("Start training...")
        startTime = datetime.datetime.now()
        classifier.fit(X_train, trainLabels)
        endTime = datetime.datetime.now()
        print("Finished training for", endTime-startTime)

        print("Starting prediction...")
        startTime = datetime.datetime.now()
        predictedLabels = classifier.predict(X_test)
        endTime = datetime.datetime.now()
        print("Finished prediction for", endTime-startTime)

        confusionMatrix = confusion_matrix(testLabels, predictedLabels)
        precision = confusionMatrix[1, 1] / (confusionMatrix[1, 1] + confusionMatrix[0, 1])
        recall = confusionMatrix[1,1] / (confusionMatrix[1,1] + confusionMatrix[1,0])
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = ((confusionMatrix[0, 0] + confusionMatrix[1, 1]) / len(predictedLabels)) * 100

        print("\nConfusion matrix:\n{0}\n".format(confusionMatrix))
        print("Precision: {0}".format(round(precision, 2)))
        print("Recall: {0}".format(round(recall, 2)))
        print("F1: {0}".format(round(f1, 2)))
        print("Accuracy: {0}".format(round(accuracy, 2)))

if __name__ == "__main__":
    train, test = train_test_split(get_dataset(), test_size=0.40)
    predictedLabels = run_classifiers(test, train)
