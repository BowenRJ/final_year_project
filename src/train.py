import csv
import pickle
import time
import sklearn
import numpy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords

def readTrainingData(vectorizedDataFile, rawDataFile):
    """
    Read the vectorized data and the target classifications.
    """

    # Read data.
    with open (vectorizedDataFile, "rb") as pickleFile:
        vectorizedData = pickle.load(pickleFile)

    # Read targets.
    target = []
    with open(rawDataFile, newline='\n', encoding="utf8") as tsvFile:
        dataReader = csv.reader(tsvFile, delimiter='\t')
        next(dataReader)    # Skip the header row.
        for row in dataReader:
            target.append(row[2])
    return vectorizedData, target


def saveModel(filename, model):
    """
    Save model from specified .pickle file.
    """

    with open(filename, "wb") as modelFile:
        pickle.dump(model, modelFile)


def printTrainingResults(y_test, y_pred, trainingTime, predictingTime):
    """
    Print results.
    """

    print("\nTime to train:", trainingTime, "\nTime to predict:", predictingTime)
    print("\nConfusion matrix:\n", sklearn.metrics.confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", sklearn.metrics.classification_report(y_test, y_pred))


plt.rcParams.update({'font.size': 14})

# Read data and target.
vectorizedData, target = readTrainingData(
    "../data/vectorizedOLID/training2.pickle", 
    "../data/OLIDv1.0/olid-training-v1.0.tsv"
)


###### May not need to permanently store tf-idf if it isn't too long.
# tf-idf (term frequency-inverse document frequency).
#vectorizedData = TfidfTransformer().fit_transform(vectorizedData).toarray()


# Split into training and testing sets.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(vectorizedData, target, test_size=0.2, random_state=0)


# Training classification model.
startTime = time.time()
# Random Forest
#randomForestClassifier = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, random_state=0)
#randomForestClassifier.fit(x_train, y_train)

# Decision Tree.
classifier = DecisionTreeClassifier(max_depth=100)
classifier.fit(x_train, y_train)

# Training time.
trainingTime = time.time() - startTime

# Predict with training set.
startTime = time.time()
#y_pred = randomForestClassifier.predict(x_test)
y_pred = classifier.predict(x_test)

predictingTime = time.time() - startTime
printTrainingResults(y_test, y_pred, trainingTime, predictingTime)

# Save model
saveModel("../models/tree0.pickle", classifier)
