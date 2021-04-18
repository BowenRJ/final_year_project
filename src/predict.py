import pickle
import time
import sklearn
import csv
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

def loadModel(filename):
    """
    Load model from specified .pickle file.
    """

    with open(filename, "rb") as modelFile:
        return pickle.load(modelFile)


def printResults(y_test, y_pred, predictingTime):
    """
    Print results.
    """

    print("\nTime to predict:", predictingTime)
    print("\nConfusion matrix:\n", sklearn.metrics.confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", sklearn.metrics.classification_report(y_test, y_pred))


def readTestData(vectorizedDataFile, rawDataFile):
    """
    Read the vectorized data and the target classifications.
    """

    # Read data.
    with open (vectorizedDataFile, "rb") as pickleFile:
        vectorizedData = pickle.load(pickleFile)

    # Read targets.
    target = []
    with open(rawDataFile, newline='\n', encoding="utf8") as csvFile:
        dataReader = csv.reader(csvFile, delimiter=',')
        for row in dataReader:
            target.append(row[1])
    return vectorizedData, target


# Read test data and target.
# data, target = readTestData(
#     "../data/OLIDv1.0/testset-levela.tsv",
#     "../data/OLIDv1.0/labels-levela.csv"
# )
vectorizedData, target = readTestData(
    "../data/vectorizedOLID/testset-levelaBagOfWords-150features_18-04-21.pickle",
    "../data/OLIDv1.0/labels-levela.csv"
)

# Load model.
classifier = loadModel("../models/tree150features_16-04-21.pickle")

# Predict.
startTime = time.time()
predictions = classifier.predict(vectorizedData)
predictingTime = time.time() - startTime

# Print results.
printResults(target, predictions, predictingTime)