import matplotlib
import numpy
import pandas
#import PyTorch
import pickle
#import Jupyter
import csv
import sklearn

def readTrainingData(filePath):
    """Read training data and returns it as a 2D array.

    Array is of the form:
    [[id, tweet string, level a, level b, level c]]
    With the levels being the classification of the tweets.
    """

    print("\nReading data from " + filePath + '\n')
    trainingData = []
    with open(filePath, newline='\n', encoding="utf8") as tsvFile:
        dataReader = csv.reader(tsvFile, delimiter='\t')
        next(dataReader)    # Skip the header row.
        for row in dataReader:
            trainingData.append(row)
    return trainingData



trainingData = readTrainingData("../data/OLIDv1.0/olid-training-v1.0.tsv")
