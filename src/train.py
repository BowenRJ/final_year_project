import csv
import pickle
import time
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import ensemble
from sklearn import neural_network
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 47
plt.rcParams.update({'font.size': 12})

def readTrainingData(vectorizedDataFile, rawDataFile):
    """
    Read the vectorized data and the target classifications.
    """

    # Read data.
    with open (vectorizedDataFile, "rb") as pickleFile:
        vectorizedData = pickle.load(pickleFile)

    # Read targets.
    target = []
    with open(rawDataFile, newline='\n', encoding="utf8") as tsvFile:
        dataReader = csv.reader(tsvFile, delimiter='\t')
        next(dataReader)    # Skip the header row.
        for row in dataReader:
            target.append(row[2])
    return vectorizedData, target


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


def encodeTarets(data):
    for i, target in enumerate(data):
        if target   == "NOT":
            data[i] = 0
        elif target == "OFF":
            data[i] = 1
        else:
            print("Error, targets not as expected.")
            exit()
    return data


def iterativelyTestAndTrain():
    maxDepth = 5
    accuracyScores = []
    maxDepths = []
    trainingTimes = []
    classificationTimes = []
    while maxDepth <= 200:
        print(str(maxDepth))
        # Training classification model.
        startTime = time.time()

        classifier = sklearn.ensemble.RandomForestClassifier(
            max_depth=None,
            n_estimators=maxDepth,
            random_state=RANDOM_STATE
        )
        classifier.fit(x_train, y_train)

        # Training time.
        trainingTime = time.time() - startTime

        # Predict with training set.
        startTime = time.time()
        #y_pred = randomForestClassifier.predict(x_test)
        y_pred = classifier.predict(x_test)

        predictingTime = time.time() - startTime
        #printTrainingResults(y_test, y_pred, trainingTime, predictingTime)

        maxDepths.append(maxDepth)
        accuracyScores.append(sklearn.metrics.f1_score(y_test, y_pred, average='macro'))
        trainingTimes.append(trainingTime)
        classificationTimes.append(predictingTime)

        maxDepth = maxDepth + 10

    plt.plot(maxDepths, accuracyScores)
    plt.title("Performance of Random Forests with Different Numbers of Trees \n (features = 3200) ", fontsize=18)
    plt.xlabel("Number of Trees", fontsize=16)
    plt.ylabel("Macro Average F1 Score", fontsize=16)
    plt.yticks(np.arange(0.6, 0.76, step=0.02))
    #plt.xticks(np.arange(50, 2000, step=50))
    plt.show()

    for i in range(len(maxDepths)):
        print(str(maxDepths[i])+"\t"+str(accuracyScores[i])+"\t"+str(trainingTimes[i])+"\t"+str(classificationTimes[i]))
    
    return None


def iterativelyTestMLP():
    print("Beginning tests...")
    classifier = sklearn.neural_network.MLPClassifier(
        #solver             = "lbfgs",
        #alpha              = 1e-5,
        #hidden_layer_sizes = (5, 2),
        max_iter     = 10000
    )

    experimentStartTime = time.time()

    parameterSpace = {
        "hidden_layer_sizes": [(200,200,200), (500,500), (500,500,500), (700), (700, 700), (1000)],
        "activation": ["relu"],#, "logistic", "tanh"],
        "solver": ["adam", "sgd"],
        "alpha": [0.000005, 0.00001, 0.0001, 0.001]
        #"learning_rate": ["constant", "adaptive"]
    }

    clf = GridSearchCV(classifier, parameterSpace, n_jobs=-1, cv=3, verbose=2)
    clf.fit(x_train, y_train)

    print(sorted(clf.cv_results_.keys()))

    # Best parameter set.
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    experimentTime = time.time() - experimentStartTime
    print("\n"+str(experimentTime))


# Read data and target.
vectorizedData, target = readTrainingData(
    "../data/vectorizedOLID/training8.pickle", 
    "../data/OLIDv1.0/olid-training-v1.0.tsv"
)


target = encodeTarets(target)


# Split into training and testing sets.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(vectorizedData, target, test_size=0.2)


#iterativelyTestMLP()


# Training classification model.
startTime = time.time()
# Random Forest.
# classifier = sklearn.ensemble.RandomForestClassifier(
#     max_depth=None,
#     n_estimators=40,
#     random_state=RANDOM_STATE
# )
# classifier.fit(x_train, y_train)

# Decision Tree.
# classifier = DecisionTreeClassifier(max_depth=40, random_state=RANDOM_STATE)
# classifier.fit(x_train, y_train)

# Neural Network.
classifier = sklearn.neural_network.MLPClassifier(
    solver             = "adam", # Which?
    activation          = "relu",
    alpha              = 0.0001,
    hidden_layer_sizes = (500,500,500),
    max_iter           = 10000 # How high?
) # dual = true for when more features than examples? (not quite true), or set to false
classifier.fit(x_train, y_train)

# Training time.
trainingTime = time.time() - startTime

# Predict with training set.
startTime = time.time()
y_pred = classifier.predict(x_test)

predictingTime = time.time() - startTime
printTrainingResults(y_test, y_pred, trainingTime, predictingTime)

# Save model.
saveModel("../models/mlp.pickle", classifier)
