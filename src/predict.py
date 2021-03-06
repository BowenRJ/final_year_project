import pickle
import time
import sklearn
import csv
import numpy
import matplotlib
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import tree
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


def encodeTarets(data):
    for i, target in enumerate(data):
        if target   == "NOT":
            data[i] = 0
        elif target == "OFF":
            data[i] = 1
        else:
            print("Error, targets not as expected.")
            exit()
    return data


# Read test data and target.
vectorizedData, target = readTestData(
    "../data/vectorizedOLID/testing8.pickle",
    "../data/OLIDv1.0/labels-levela.csv"
)

target = encodeTarets(target)

# Load model.
classifier = loadModel("../models/mlp.pickle")

# Predict.
startTime = time.time()
predictions = classifier.predict(vectorizedData)
predictingTime = time.time() - startTime

# Print results.
printResults(target, predictions, predictingTime)

# print(classifier.tree_.max_depth)
# print(classifier.tree_.n_leaves)

#tree.plot_tree(classifier)
#pyplot.show()