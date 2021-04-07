import matplotlib
import numpy
import pandas
import pickle
import csv
import sklearn
import regex
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier

def readTrainingData(filePath):
    """
    Read training data and returns it as a 2D array.

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

def cleanseData(data):
    """
    Cleanse data. 
    
    Remove everything but alphanumeric characters and then lemmatize the remaining words.
    """
    processedData = []
    stemmer = WordNetLemmatizer()
    for tweet in range(0, len(data)):
        # Remove "@USER" instances.
        processedTweet = regex.sub(r'@USER', ' ', str(data[tweet]))

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
        processedTweet = processedTweet.split()

        processedTweet = [stemmer.lemmatize(word) for word in processedTweet]
        processedTweet = ' '.join(processedTweet)   # Convert back to string.
        
        processedData.append(processedTweet)
    return processedData    

trainingData = readTrainingData("../data/OLIDv1.0/olid-training-v1.0.tsv")
# Separate data and target.
data   = [i[1] for i in trainingData]   # data[1] = tweet
target = [i[2] for i in trainingData]   # data[2] = classification

print("Tweet 1: ", data[1], "\nClassification: ", target[1])

processedData = cleanseData(data)    

print("\n", processedData[1])

# Convert data to bag of words.
vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
vectorizedData = vectorizer.fit_transform(processedData).toarray()
print(vectorizedData)

#TFIDF WOULD BE HERE...

exit()

# Split into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(vectorizedData, target, test_size=0.2, random_state=0)

# Training classification model
startTime = time.time()
# Random Forest
#randomForestClassifier = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, random_state=0)
#randomForestClassifier.fit(x_train, y_train)

# Decision Tree
treeClassifier = DecisionTreeClassifier()
treeClassifier.fit(x_train, y_train)

trainingTime = time.time() - startTime()

# Predict
startTime = time.time()
#y_pred = randomForestClassifier.predict(x_test)
y_pred = treeClassifier.predict(x_test)

predictingTime = time.time() - startTime()

# Resuts
print("\nTime to train:", trainingTime, "\nTime to predict:", predictingTime)
print("\nConfusion matrix:", sklearn.metrics.confusion_matrix(y_test, y_pred))
#print(sklearn.metrics.confusion_matrix(y_test, y_pred))
print("\nClassification report:", sklearn.metrics.classification_report(y_test, y_pred))
#print(sklearn.metrics.classification_report(y_test, y_pred))

# Save model