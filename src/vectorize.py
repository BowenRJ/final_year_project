import csv
import sklearn
import numpy
import pickle
import time
from nltk.corpus import stopwords

def readCleansedData(filepath):
    """
    Read cleansed data.
    """

    print("Reading cleansed data from " + filepath)

    with open (filepath, "rb") as pickleFile:
        cleansedData = pickle.load(pickleFile)
    return cleansedData


def saveVectorizedData(data, filepath):
    """
    Write vectorized data to a specified file location.
    """

    print("Writing vectorized data to " + filepath)
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


filenameSuffix = "5"
startTime = time.time()

# Read cleansed data.
cleansedTrainingData = readCleansedData("../data/cleansedOLID/training1.pickle")
cleansedTestingData  = readCleansedData("../data/cleansedOLID/testing1.pickle")
cleansedData = cleansedTrainingData + cleansedTestingData


# Create count vectorizer on all present data.
vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        max_features=1600,
        min_df=5,            # Minimum document frequency.
        max_df=0.7,          # Maximum document frequency (%).
        stop_words=stopwords.words('english')
)
vectorizer = vectorizer.fit(cleansedData)

# tf-idf (term frequency-inverse document frequency).
#vectorizedData = TfidfTransformer().fit_transform(vectorizedData).toarray()

# Vectorize training and testing data.
#vectorizedData = vectorizer.fit_transform(cleansedData).toarray()
vectorizedTrainingData = vectorizer.transform(cleansedTrainingData) 
vectorizedTestingData  = vectorizer.transform(cleansedTestingData) 


# Save data.
# This may be dependant on method of vectorizing? Let's wait till all implemented,
# And then can consider making these separate function things.
saveVectorizedData(vectorizedTrainingData, "../data/vectorizedOLID/training"+filenameSuffix+".pickle")
saveVectorizedData(vectorizedTestingData,  "../data/vectorizedOLID/testing"+filenameSuffix+".pickle")

# Print time.
endTime = time.time()
vectorizingTime = endTime - startTime
print("Time taken to vectorize "+str(len(cleansedData))+" items of training/testing data: "+str(vectorizingTime))
