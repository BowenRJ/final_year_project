import csv
import regex
import pickle
import time
from nltk.stem import WordNetLemmatizer
 

def readRawData(filepath):
    """
    Read raw data and returns it as a 2D array.

    Array is of the form:
    [[id, tweet string, level a, level b, level c]]
    With the levels being the classification of the tweets.
    """

    print("Reading raw data from " + filepath)
    rawData = []
    with open(filepath, newline='\n', encoding="utf8") as tsvFile:
        dataReader = csv.reader(tsvFile, delimiter='\t')
        next(dataReader)    # Skip the header row.
        for row in dataReader:
            rawData.append(row)
    return rawData


def cleanseData(data):
    """
    Cleanse data. 
    
    Remove everything but alphanumeric characters and then lemmatize the remaining words.
    """

    cleansedData = []
    stemmer = WordNetLemmatizer()
    for i in range(0, len(data)):
        processedTweet = data[i][1]
        # Remove "@USER" instances.
        processedTweet = regex.sub(r'@USER', ' ', processedTweet)

        # Remove "URL" instances.
        processedTweet = regex.sub(r'URL', ' ', processedTweet)

        # Remove any non-alphanumeric characters.
        processedTweet = regex.sub(r'\W', ' ', processedTweet)
        
        # Remove any single character surrounded by whitespace.
        processedTweet = regex.sub(r'\s+[a-zA-Z]\s+', ' ', processedTweet)
        
        # Remove single characters at the start which are follow by whitespace.
        processedTweet = regex.sub(r'\^[a-zA-Z]\s+', ' ', processedTweet) 
        
        # Remove multiple consecutive whitespaces.
        processedTweet = regex.sub(r'\s+', ' ', processedTweet) ###### REMOVED flag=re.I (ignore case)
        
        # Convert to Lowercase.
        processedTweet = processedTweet.lower()
        
        # Lemmatize tweet.
        processedTweet = processedTweet.split()    # Convert to list.
        processedTweet = [stemmer.lemmatize(word) for word in processedTweet]
        processedTweet = ' '.join(processedTweet)    # Convert back to string.
        
        cleansedData.append(processedTweet)
    return cleansedData    


def saveCleansedData(data, filepath):
    """
    Write raw data to a specified file location.
    """

    print("Writing cleansed data to " + filepath)
    with open(filepath, "wb") as pickleFile:
        pickle.dump(data, pickleFile)


filenameSuffix = "0"

# Cleanse training data.
startTime = time.time()
rawData = readRawData("../data/OLIDv1.0/olid-training-v1.0.tsv")
cleansedData = cleanseData(rawData)
saveCleansedData(cleansedData, "../data/cleansedOLID/training"+filenameSuffix+".pickle")
endTime = time.time()
cleansingTime = endTime - startTime
print("Time taken to cleanse "+str(len(rawData))+" items of training data: "+str(cleansingTime)+"\n")

# Cleanse testing data.
startTime = time.time()
rawData = readRawData("../data/OLIDv1.0/testset-levela.tsv")
cleansedData = cleanseData(rawData)
saveCleansedData(cleansedData, "../data/cleansedOLID/testing"+filenameSuffix+".pickle")
endTime = time.time()
cleansingTime = endTime - startTime
print("Time taken to cleanse "+str(len(rawData))+" items of testing data: "+str(cleansingTime))
