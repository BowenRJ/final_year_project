import matplotlib
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 12})

def validationVsTesting_tree():
    # Grouped bar for validation vs testing:
    labels       = ["tree1", "tree2", "tree3", "tree4"]
    validationF1 = [0.61, 0.64, 0.67, 0.67]
    testingF1    = [0.63, 0.66, 0.70, 0.69]
    x            = np.arange(len(labels))
    width        = 0.25

    fig, ax = plt.subplots()
    rects1  = ax.bar(x - width/2, validationF1, width, label="Validation")
    rects2  = ax.bar(x + width/2, testingF1, width, label="Testing")

    ax.set_ylabel("Macro Average F1 Score", fontsize=16)
    ax.set_xlabel("Model", fontsize=16)
    ax.set_title("Comparison of Validation and Testing Scores", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels) # fontsize=default
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


def executionTimes(): # 14100 training tweets, 11280 for training, 2820 validation
    featureNumbers = [5, 15, 15, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195]
    #times          = [1.09,2.83,5.12,5.27,5.51,8.26,6.43,7.21,8.12,8.87,9.92,10.93,11.67,12.76,13.66,14.76,15.84,23.17,27.93,26.04] # 3200 features
    times           = [0.58,1.27,2.21,3.01,4.51,8.05,7.69,8.68,9.62,10.94,12.79,13.48,14.3,15.95,16.97,18.27,19.03,20.67,22.11,25.69] # 1600 features

    plt.plot(featureNumbers, times)
    plt.title("Training Times of Random Forests \n With 1600 Features", fontsize=18)
    plt.xlabel("Number of Trees", fontsize=16)
    plt.ylabel("Time to Train (Seconds)", fontsize=16)
    plt.yticks(np.arange(0, 28, step=4))
    #plt.xticks(np.arange(50, 2000, step=50))
    plt.show()



#validationVsTesting_tree()

executionTimes()