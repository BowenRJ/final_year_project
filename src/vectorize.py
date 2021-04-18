import csv
import sklearn
import numpy
import pickle
from nltk.corpus import stopwords

def readCleansedData(filepath):
    """
    Read cleansed data.
    """

    print("\nReading data from " + filepath + '\n')

    with open (filepath, "rb") as pickleFile:
        cleansedData = pickle.load(pickleFile)
    return cleansedData


def saveVectorizedData(data, filepath):
    """
    Write vectorized data to a specified file location.
    """
    print("\nWriting data to " + filepath + '\n')
    with open(filepath, "wb") as pickleFile:
        pickle.dump(data, pickleFile)


def bagOfWords(data):
    """
    Convert text data to numeric by use of the bag of words approach.
    """


def Word2Vec(data):
    print("not implemented")  


def BERT(data):
    print("not implemented")


# Read cleansed data.
#cleansedData = readCleansedData("../data/cleansedOLID/testset-levela_18-04-21.pickle")
cleansedData = readCleansedData("../data/cleansedOLID/training_08-04-21.pickle")


# Vectorize data by bag of words method.
vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        max_features=150,
        min_df=5,
        max_df=0.7,
        stop_words=stopwords.words('english')
)
vectorizedData = vectorizer.fit_transform(cleansedData).toarray()


# tf-idf (term frequency-inverse document frequency).
#vectorizedData = TfidfTransformer().fit_transform(vectorizedData).toarray()


# Save data.
# This may be dependant on method of vectorizing? Let's wait till all implemented,
# And then can consider making these separate function things.

#saveVectorizedData(vectorizedData, "../data/vectorizedOLID/testset-levelaBagOfWords-150features_18-04-21.pickle")
saveVectorizedData(vectorizedData, "../data/vectorizedOLID/trainingBagOfWords-150features_08-04-21.pickle")

