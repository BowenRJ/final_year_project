import csv
import regex
from nltk.stem import WordNetLemmatizer

def readRawData(filepath):
    """
    Read raw data and returns it as a 2D array.

    Array is of the form:
    [[id, tweet string, level a, level b, level c]]
    With the levels being the classification of the tweets.
    """

    print("\nReading data from " + filepath + '\n')
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
        
        cleansedData.append([data[i][0], processedTweet, data[i][2], data[i][3], data[i][4]])
    return cleansedData    

def saveCleansedData(data, filepath):
    """
    Write raw data to a specified file location.

    """

    print("\nWriting data to " + filePath + '\n')
    with open(filePath, "w", newline='\n', encoding="utf8") as tsvFile:
        dataWriter = csv.writer(tsvFile, delimiter='\t')
        dataWriter.writerows(data)

### MAKE FUNCTION FOR CREATING NUMERIC DATA, AND SAVING IT


rawData = readRawData("../data/OLIDv1.0/olid-training-v1.0.tsv")
print(rawData[1])
cleansedData = cleanseData(rawData)
print("Tweet 1: ", cleansedData[1][1], "\nClassification: ", cleansedData[1][2])
saveCleansedData(cleansedData, "../data/cleansedOLID/training08-04-21.tsv")
