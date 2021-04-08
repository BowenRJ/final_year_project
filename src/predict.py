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

def readTestData(filen):
    print("to implement")

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

####### CURRENTLY JUST USING TRAINING DATA.

# Read data and target.
data, target = readTrainingData("../data/cleansedOLID/training08-04-21.tsv")

# Convert data to bag of words.
vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
vectorizedData = vectorizer.fit_transform(data).toarray()

# Split data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(vectorizedData, target, test_size=0.2, random_state=0)

# Load model
classifier = loadModel("../models/tree07-04-21.pickle")

# Predict
startTime = time.time()
y_pred = classifier.predict(x_test)

predictingTime = time.time() - startTime

printResults(y_test, y_pred, predictingTime)