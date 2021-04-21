import pickle
import time
import sklearn
import csv
import numpy
import matplotlib
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.tree import DecisionTreeClassifier
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


# Read test data and target.
vectorizedData, target = readTestData(
    "../data/vectorizedOLID/testing2.pickle",
    "../data/OLIDv1.0/labels-levela.csv"
)

# Load model.
classifier = loadModel("../models/tree3.pickle")

# Predict.
startTime = time.time()
predictions = classifier.predict(vectorizedData)
predictingTime = time.time() - startTime

# Print results.
printResults(target, predictions, predictingTime)

print(classifier.tree_.max_depth)

tree.plot_tree(classifier)
pyplot.show()