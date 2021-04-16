import csv
import pickle
import time
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords

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

# Read data and target.
data, target = readTrainingData("../data/cleansedOLID/training08-04-21.tsv")

# Convert data to bag of words.
vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    max_features=150,    # Note: test set can only provide 421 features.
    min_df=5, 
    max_df=0.7, 
    stop_words=stopwords.words('english')
)
vectorizedData = vectorizer.fit_transform(data).toarray()

# tf-idf (term frequency-inverse document frequency).
#vectorizedData = TfidfTransformer().fit_transform(vectorizedData).toarray()


# Split into training and testing sets.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(vectorizedData, target, test_size=0.2, random_state=0)


# Training classification model.
startTime = time.time()
# Random Forest
#randomForestClassifier = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, random_state=0)
#randomForestClassifier.fit(x_train, y_train)

# Decision Tree
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

trainingTime = time.time() - startTime

# Predict with training set
startTime = time.time()
#y_pred = randomForestClassifier.predict(x_test)
y_pred = classifier.predict(x_test)

predictingTime = time.time() - startTime
printTrainingResults(y_test, y_pred, trainingTime, predictingTime)

# Save model
saveModel("../models/tree150features_16-04-21.pickle", classifier)
