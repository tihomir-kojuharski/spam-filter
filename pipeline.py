from csv import DictReader

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# from features.counting_feat import SentenceLength, BagOfTfIDF, WordOverlap
# from features.nltk_feat import POS, NER
# from features.preprocess import TokenizedLemmas

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
        ('word_counts', WordCounts(train))
        # ('preprocess_lemmas', TokenizedLemmas()),
        # ('sent_len', SentenceLength()),
        # ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
        # ('pos', POS()),
        # ('ner', NER()),
        # ('word_overlap', WordOverlap())
    ]
    return features


def get_pipeline(features, clf):
    feature_names = []
    for feature in features:
        feature_names += feature[1].FEATS
    print(feature_names)
    return Pipeline(features + [('transform', ToMatrix(features=feature_names)),
                                ('norm', MinMaxScaler())])


def run_classifier(test, train):
    clf = SVC(kernel="linear", C=0.025)

    # print(test)

    #  get pipeline of features and generate them
    pipeline = get_pipeline(get_features(train), clf)
    X_train = pipeline.fit_transform(train)

    print("Start training...")
    true_labels = [instance['is_spam'] for instance in train]
    clf.fit(X_train, true_labels)
    print("Finished training...")

    # get the features for the test set
    X = pipeline.fit_transform(test)
    # return the predicted labels of the test set
    return clf.predict(X)


if __name__ == "__main__":
    # test_cross_validation()
    dataset = get_dataset()

    train, test = train_test_split(dataset, test_size=0.40)

    predictedLabels = run_classifier(test, train)
    goldLabels = [instance["is_spam"] for instance in test]

    print("==================\n{0}:\n{1}".format(str("SVM"), confusion_matrix(goldLabels, predictedLabels)))