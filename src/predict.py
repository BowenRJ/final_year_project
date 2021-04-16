import pickle
import time
import sklearn
import csv
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

def readTestData(testData, testTargets):
    """
    Read test data from 2 files, and return as 2 separate lists, the data and the targets.
    """
    # Read data.
    data = []
    with open(testData, newline='\n', encoding="utf8") as tsvFile:
        dataReader = csv.reader(tsvFile, delimiter='\t')
        next(dataReader)    # Skip the header row.
        for row in dataReader:
            #print(row)
            data.append(row[1])
    # Read target.
    target = []
    with open(testTargets, newline='\n', encoding="utf8") as csvFile:
        dataReader = csv.reader(csvFile, delimiter=',')
        for row in dataReader:
            #print(row)
            target.append(row[1])
            #print(row[1])
    return data, target

def readTrainingData(filepath):
    data   = []
    target = []
    with open(filepath, newline='\n', encoding="utf8") as tsvFile:
        dataReader = csv.reader(tsvFile, delimiter='\t')
        next(dataReader)    # Skip the header row.
        for row in dataReader:
            data.append(row[1])
            target.append(row[2])
    return data, target


# Read test data and target.
data, target = readTestData("../data/OLIDv1.0/testset-levela.tsv", "../data/OLIDv1.0/labels-levela.csv")

# Convert data to bag of words.
vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    max_features=150,
    min_df=5,
    max_df=0.7,
    stop_words=stopwords.words('english')
)
vectorizedData = vectorizer.fit_transform(data).toarray()

# Load model
classifier = loadModel("../models/tree150features_16-04-21.pickle")

# Predict
startTime = time.time()
predictions = classifier.predict(vectorizedData)

predictingTime = time.time() - startTime

printResults(target, predictions, predictingTime)