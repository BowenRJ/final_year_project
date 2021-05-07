import csv
import sklearn
import numpy
import pickle
import time
from nltk.corpus import stopwords

# from gensim.test.utils import common_texts # More?
# from gensim.models import Word2Vec
# from gensim.models.Doc2Vec import Doc2Vec, TaggedDocument

#import gensim

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


def bagOfWords(cleansedData):
    """
    Convert text data to numeric by use of the bag of words approach.
    """

    # Create count vectorizer on all data.
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            max_features=6400,
            min_df=1,            # Minimum document frequency.
            max_df=0.7,          # Maximum document frequency (%).
            stop_words=stopwords.words('english')
    )
    vectorizer = vectorizer.fit(cleansedData)

    # Vectorize training and testing data.
    vectorizedTrainingData = vectorizer.transform(cleansedTrainingData)
    vectorizedTestingData  = vectorizer.transform(cleansedTestingData)

    return vectorizedTrainingData, vectorizedTestingData


def tfidf(cleansedData):
    """
    Convert text data to numeric by use of the tf-idf approach.
    """

    # Create tfidf vectorizer on all data.
    tfidfVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_features=6400, min_df=2) # initialize
    tfidfVectorizer = tfidfVectorizer.fit(cleansedData)

    # Vectorize training and testing data.
    vectorizedTrainingData = tfidfVectorizer.transform(cleansedTrainingData) 
    vectorizedTestingData  = tfidfVectorizer.transform(cleansedTestingData)

    return vectorizedTrainingData, vectorizedTestingData


def Word2Vecectorize(cleansedData):
    """
    Convert text data to numeric by use of the Word2Vec model.
    """

    # Split each tweet into a list of words.
    for i in range(len(cleansedData)):
        cleansedData[i] = cleansedData[i].split(" ")

    print(cleansedData[0])

    print("Beginning Word2Vec...")

    # model = Doc2Vec(
    #     documents = cleansedData,
    #     size      = 100,
    #     min_count = 2, # Change?
    #     dm        = 1
    # )
    # model.build_vocab()
    # word2VecModel.save("model1.bin")

    print(word2VecModel)

    print(word2VecModel.wv.most_similar("awful"))
    


    # Vectorize training.



    # Vectorize testing.



    #return vectorizedTrainingData, vectorizedTestingData


def BERT(data):
    print("not implemented")


filenameSuffix = "17"
startTime = time.time()

# Read cleansed data.
cleansedTrainingData = readCleansedData("../data/cleansedOLID/training1.pickle")
cleansedTestingData  = readCleansedData("../data/cleansedOLID/testing1.pickle")
cleansedData = cleansedTrainingData + cleansedTestingData

# Vectorize data.
vectorizedTrainingData, vectorizedTestingData = bagOfWords(cleansedData)
#vectorizedTrainingData, vectorizedTestingData = tfidf(cleansedData)

print(vectorizedTrainingData)
#Word2Vecectorize(cleansedData)

#exit()

# Save data.
# This may be dependant on method of vectorizing? Let's wait till all implemented,
# And then can consider making these separate function things.
##saveVectorizedData(vectorizedTrainingData, "../data/vectorizedOLID/training"+filenameSuffix+".pickle")
##saveVectorizedData(vectorizedTestingData,  "../data/vectorizedOLID/testing"+filenameSuffix+".pickle")

# Print time.
endTime = time.time()
vectorizingTime = endTime - startTime
print("Time taken to vectorize "+str(len(cleansedData))+" items of training/testing data: "+str(vectorizingTime))
