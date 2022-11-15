

"""
BBM409 Introduction to Machine Learning Lab. Fall 2022.
Assignment 1: PART 1 : Personality Classification

Contributors:
Ali Argun Sayilgan : 21827775
Mehmet Giray Nacakci : 21989009
"""

import pandas as pd
import numpy as np
import random
import time
from IPython.display import display
pd.set_option('display.precision', 6)
pd.set_option('display.max_columns', None)


# import and preprocess the dataset
df = pd.read_csv("../subset_16P.csv", encoding='cp1252')
numberOfClasses = len(df["Personality"].unique())
personality_types = ["ESTJ", "ENTJ", "ESFJ", "ENFJ", "ISTJ", "ISFJ", "INTJ", "INFJ", "ESTP", "ESFP", "ENTP", "ENFP", "ISTP", "ISFP", "INTP", "INFP"]
df.Personality = df.Personality.astype("category", personality_types).cat.codes


# df["Personality"] = pd.Categorical(df["Personality"])
# df["Personality"] = df["Personality"].cat.codes

X = df.drop(['Response Id', 'Personality'], axis=1)
y = df.Personality
X = X.to_numpy()
Y = y.to_numpy()


# feature normalization
class MinMaxScaler():
    def __init__(self):
        self.mins = []
        self.maxes = []

    def fit_transform(self, X):
        self.mins = X.min(axis=0)
        self.maxes = X.max(axis=0)
        maxMinusMin = self.maxes - self.mins
        return (X - self.mins) / maxMinusMin

    def transform(self, X):
        maxMinusMin = self.maxes - self.mins
        return (X - self.mins) / maxMinusMin


class KFold():
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.shuffle = shuffle
        self.n_splits = n_splits
        self.random_state = random_state

    # Fisher-Yates Shuffle Algorithm
    def shuffler(self, arr, n):
        random.seed(n)
        rowSize = arr.shape[0]
        for i in range(rowSize - 1, 0, -1):
            # random index from 0 to i
            j = random.randint(0, i + 1)

            # Swap with random index
            arr[[i, j]] = arr[[j, i]]
        return arr

    def split(self, X, y):
        if (self.shuffle):
            X = self.shuffler(X, self.random_state)
            y = self.shuffler(y, self.random_state)

        rowSize = len(X)
        testSetSize = rowSize // self.n_splits
        for i in range(self.n_splits):
            if (i == 0):
                x_train = X[(i + 1) * testSetSize:, ]
                y_train = Y[(i + 1) * testSetSize:, ]
            elif (i == self.n_splits - 1):
                x_train = X[:i * testSetSize, ]
                y_train = Y[:i * testSetSize, ]
            else:
                # [ row1,row2, ..., x_train_rows, rowk, ...]
                # appending rows prior to x_train with rows comes after x_train
                x_train_smaller_indices = X[:i * testSetSize, ]
                y_train_smaller_indices = Y[:i * testSetSize, ]
                x_train = np.append(
                    x_train_smaller_indices, X[(i + 1) * testSetSize:, ], axis=0
                )
                y_train = np.append(
                    y_train_smaller_indices, Y[(i + 1) * testSetSize:, ], axis=0
                )

            if (i != self.n_splits - 1):
                x_test = X[i * testSetSize: (i + 1) * testSetSize, ]
                y_test = Y[i * testSetSize: (i + 1) * testSetSize, ]
            else:
                #           because we calculate testSetSize with //,
                #           last split must finish through the end of the whole array
                x_test = X[i * testSetSize:, ]
                y_test = Y[i * testSetSize:, ]
            yield (x_train, x_test, y_train, y_test)


class KNNClassifier():
    def __init__(self, n_neighbors=5, weights='uniform', n_classes=16):

        self.X_train = None
        self.y_train = None

        self.n_classes = n_classes
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidian_distance(self, a, b):
        distances = np.sqrt(np.sum((a - b) ** 2, axis=1))
        # prevent division by zero
        distances[np.where(distances < 0.00001)] = 0.00001
        return distances

    def kneighbors(self, X_test, return_distance=False):

        dist = []
        neigh_ind = []

        point_dist = [self.euclidian_distance(x_test, self.X_train) for x_test in X_test]

        for row in point_dist:
            enum_neigh = enumerate(row)
            sorted_neigh = sorted(enum_neigh,
                                  key=lambda x: x[1])[:self.n_neighbors]

            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]

            dist.append(dist_list)
            neigh_ind.append(ind_list)

        if return_distance:
            return np.array(dist), np.array(neigh_ind)

        return np.array(neigh_ind)

    def predict(self, X_test):

        # non-weighted knn, majority voting of neighbors for classification
        if self.weights == 'uniform':
            neighbors = self.kneighbors(X_test)
            y_pred = np.array([
                np.argmax(np.bincount(self.y_train[neighbor]))
                for neighbor in neighbors
            ])
            return y_pred

        # weighted knn, voting based on weights of neighbors
        if self.weights == 'distance':

            dist, neigh_ind = self.kneighbors(X_test, return_distance=True)

            inv_dist = 1 / dist

            mean_inv_dist = inv_dist / np.sum(inv_dist, axis=1)[:, np.newaxis]

            proba = []

            for i, row in enumerate(mean_inv_dist):

                row_pred = self.y_train[neigh_ind[i]]

                for k in range(self.n_classes):
                    indices = np.where(row_pred == k)
                    prob_ind = np.sum(row[indices])
                    proba.append(np.array(prob_ind))

            predict_proba = np.array(proba).reshape(X_test.shape[0],
                                                    self.n_classes)

            y_pred = np.array([np.argmax(item) for item in predict_proba])

            return y_pred

        # used for interpretation of misclassified samples, return also nearest neighbors
        elif self.weights == 'uniform_neighbors':
            neighbors = self.kneighbors(X_test)  # nearestNeighborsIndices_of_all_testSamples
            y_pred = np.array([
                np.argmax(np.bincount(self.y_train[neighbor]))
                for neighbor in neighbors
            ])
            return y_pred, neighbors



class Pipeline():
    def __init__(self, scaler=None, classifier=None):
        self.scaler = scaler
        self.classifier = classifier

    def execute(self, x_train, x_test, y_train):
        if (self.scaler is not None):
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)
        if (self.classifier is not None):
            self.classifier.fit(x_train, y_train)
            return self.classifier.predict(x_test)


""" Classification Metrics """

def accuracy(pred, actual):
    return sum(pred == actual) / len(pred)


def precision(pred, actual):
    if (len(pred) == 0 or len(pred) != len(actual)):
        return -1
    labels = []
    truePositivesPerLabel = {}
    falsePositivesPerLabel = {}
    precisionPerLabel = {}

    for i in range(len(pred)):
        prediction = pred[i]
        if prediction not in labels:
            labels.append(prediction)
            truePositivesPerLabel[prediction] = 0
            falsePositivesPerLabel[prediction] = 0

        if (pred[i] == actual[i]):
            truePositivesPerLabel[prediction] += 1
        else:
            falsePositivesPerLabel[prediction] += 1

    # count of the labels that are existed inside the ground truth or prediction
    existedLabelCount = 0

    precisionSum = 0
    for label in labels:
        denominator = truePositivesPerLabel[label] + falsePositivesPerLabel[label]
        if (denominator >= 0):
            existedLabelCount += 1
            precisionSum += truePositivesPerLabel[label] / denominator

    return precisionSum / existedLabelCount


def recall(pred, actual):
    if (len(pred) == 0 or len(pred) != len(actual)):
        return -1
    labels = []
    truePositivesPerLabel = {}
    falseNegativesPerLabel = {}
    recallPerLabel = {}

    for i in range(len(actual)):
        actualClass = actual[i]
        if actualClass not in labels:
            labels.append(actualClass)
            truePositivesPerLabel[actualClass] = 0
            falseNegativesPerLabel[actualClass] = 0

        if (pred[i] == actual[i]):
            truePositivesPerLabel[actualClass] += 1
        else:
            falseNegativesPerLabel[actualClass] += 1

    # count of the labels that are existed inside the ground truth or prediction
    existedLabelCount = 0

    recallSum = 0
    for label in labels:
        denominator = truePositivesPerLabel[label] + falseNegativesPerLabel[label]
        if (denominator >= 0):
            existedLabelCount += 1
            recallSum += truePositivesPerLabel[label] / denominator

    return recallSum / existedLabelCount


def cross_val_score(X, Y, cv, pipeline):

    accuracy_folds = []
    precision_folds = []
    recall_folds = []

    # for each Fold of 5-fold-validation
    for (x_train, x_test, y_train, y_test) in cv.split(X,Y):
        y_pred = pipeline.execute(x_train, x_test, y_train)

        accuracy_folds.append(accuracy(y_pred, y_test))
        precision_folds.append(recall(y_pred, y_test))
        recall_folds.append(precision(y_pred, y_test))

    # averages of folds
    accuracy_folds.append(sum(accuracy_folds)/5)
    precision_folds.append(sum(precision_folds)/5)
    recall_folds.append(sum(recall_folds)/5)

    return accuracy_folds, precision_folds, recall_folds




""" ********  Run non-weighted and weighted KNN models  (20 variations)  ******** """

cv = KFold(5, shuffle=True, random_state=24)
scaler = MinMaxScaler()
neighborVariations = [1,3,5,7,9]

accuracy_table_columns = []
precision_table_columns = []
recall_table_columns = []


def run_all_models():
    print(" \nResults of 20 KNN model variations will be ready after ABOUT  25  MINUTES  of execution. Please wait... \n")
    progress = 1
    start = time.time()

    """  ***   NON-WEIGHTED KNN   *** """
    for k in neighborVariations:   # THIS LOOP TAKES ABOUT 10 MINUTES TO COMPLETE

        knnUniform = KNNClassifier(n_neighbors=k, weights='uniform', n_classes=numberOfClasses)

        # with feature normalization
        print("  KNN model variation no:   " + str(progress) + "  is started being processed..." )
        pipeline = Pipeline(scaler=scaler, classifier=knnUniform)
        accuracies, precisions, recalls = cross_val_score(X, Y, cv, pipeline)
        accuracy_table_columns.append(accuracies)
        precision_table_columns.append(precisions)
        recall_table_columns.append(recalls)
        print("   KNN model variation no:  " + str(progress) + "  processing is finished.\n" )
        progress += 1

        # without feature normalization
        print("  KNN model variation no:   " + str(progress) + "  is started being processed..." )
        pipeline = Pipeline(classifier=knnUniform)
        accuracies, precisions, recalls = cross_val_score(X, Y, cv, pipeline)
        accuracy_table_columns.append(accuracies)
        precision_table_columns.append(precisions)
        recall_table_columns.append(recalls)
        print("   KNN model variation no:  " + str(progress) + "  processing is finished.\n" )
        progress += 1


    """  ***   WEIGHTED KNN   *** """
    for k in neighborVariations:   # THIS LOOP TAKES ABOUT 15 MINUTES TO COMPLETE

        knnDistance = KNNClassifier(n_neighbors=5, weights='distance', n_classes=numberOfClasses)

        # with feature normalization
        print("  KNN model variation no:   " + str(progress) + "  is started being processed..." )
        pipeline = Pipeline(scaler=scaler, classifier=knnDistance)
        accuracies, precisions, recalls = cross_val_score(X, Y, cv, pipeline)
        accuracy_table_columns.append(accuracies)
        precision_table_columns.append(precisions)
        recall_table_columns.append(recalls)
        print("   KNN model variation no:  " + str(progress) + "  processing is finished.\n" )
        progress += 1

        # without feature normalization
        print("  KNN model variation no:   " + str(progress) + "  is started being processed..." )
        pipeline = Pipeline(classifier=knnDistance)
        accuracies, precisions, recalls = cross_val_score(X, Y, cv, pipeline)
        accuracy_table_columns.append(accuracies)
        precision_table_columns.append(precisions)
        recall_table_columns.append(recalls)
        print("   KNN model variation no:   " + str(progress) + "  processing is finished.\n" )
        progress += 1


    # model calculations are finished.
    finish = time.time()
    seconds = finish-start
    minutes = seconds//60
    seconds -= 60*minutes
    print("Results of 20 KNN model variations are ready in the sections below. Thank you for your patience.")
    print('Elapsed time is:   %d:%d   minutes:seconds\n' %(minutes,seconds))


run_all_models()





""" Cross Validation Scores (Tables will be ready after about 25 minutes of execution) """

def draw_accuracy_table():
    print("------ Accuracy - for 20 model variations ------")
    accuracy_rows = np.transpose(np.array(accuracy_table_columns))
    accuracy_table = pd.DataFrame(accuracy_rows, columns = ['1: k=1 w- n+','2: k=1 w- n-','3: k=3 w- n+','4: k=3 w- n-','5: k=5 w- n+','6: k=5 w- n-','7: k=7 w- n+','8: k=7 w- n-','9: k=9 w- n+','10: k=9 w- n-','11: k=1 w+ n+','12: k=1 w+ n-','13: k=3 w+ n+','14: k=3 w+ n-','15: k=5 w+ n+','16: k=5 w+ n-','17: k=7 w+ n+','18: k=7 w+ n-','19: k=9 w+ n+','20: k=9 w+ n-'])
    accuracy_table.index = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average of Folds']
    display(accuracy_table.iloc[:, :10].head(6))
    display(accuracy_table.iloc[:, 10:].head(6))
    print("\nmodel variations encoding: \n k=  : k parameter of KNN \n w+  :       weighted KNN \n w-  :   non-weighted KNN \n n+  :    with feature normalization \n n-  : without feature normalization \n")
draw_accuracy_table()

def draw_precision_table():
    print("------ Precision - for 20 model variations ------")
    precision_rows = np.transpose(np.array(precision_table_columns))
    precision_table = pd.DataFrame(precision_rows, columns = ['1: k=1 w- n+','2: k=1 w- n-','3: k=3 w- n+','4: k=3 w- n-','5: k=5 w- n+','6: k=5 w- n-','7: k=7 w- n+','8: k=7 w- n-','9: k=9 w- n+','10: k=9 w- n-','11: k=1 w+ n+','12: k=1 w+ n-','13: k=3 w+ n+','14: k=3 w+ n-','15: k=5 w+ n+','16: k=5 w+ n-','17: k=7 w+ n+','18: k=7 w+ n-','19: k=9 w+ n+','20: k=9 w+ n-'])
    precision_table.index = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average of Folds']
    display(precision_table.iloc[:, :10].head(6))
    display(precision_table.iloc[:, 10:].head(6))
    print("\nmodel variations encoding: \n k=  : k parameter of KNN \n w+  :       weighted KNN \n w-  :   non-weighted KNN \n n+  :    with feature normalization \n n-  : without feature normalization \n")
draw_precision_table()

def draw_recall_table():
    print("------ Recall - for 20 model variations ------")
    recall_rows = np.transpose(np.array(recall_table_columns))
    recall_table = pd.DataFrame(recall_rows, columns = ['1: k=1 w- n+','2: k=1 w- n-','3: k=3 w- n+','4: k=3 w- n-','5: k=5 w- n+','6: k=5 w- n-','7: k=7 w- n+','8: k=7 w- n-','9: k=9 w- n+','10: k=9 w- n-','11: k=1 w+ n+','12: k=1 w+ n-','13: k=3 w+ n+','14: k=3 w+ n-','15: k=5 w+ n+','16: k=5 w+ n-','17: k=7 w+ n+','18: k=7 w+ n-','19: k=9 w+ n+','20: k=9 w+ n-'])
    recall_table.index = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average of Folds']
    display(recall_table.iloc[:, :10].head(6))
    display(recall_table.iloc[:, 10:].head(6))
    print("\nmodel variations encoding: \n k=  : k parameter of KNN \n w+  :       weighted KNN \n w-  :   non-weighted KNN \n n+  :    with feature normalization \n n-  : without feature normalization \n")
draw_recall_table()



""" Displaying Misclassified samples """

print(" \n3 misclassified samples and their neighbors will be ready in about  1  MINUTE  of execution. Please wait... \n")
cv = KFold(5,shuffle=True, random_state=24)
X_train, X_test, y_train, y_test = next(cv.split(X,Y))
knnUniform = KNNClassifier(n_neighbors=7, weights='uniform_neighbors', n_classes=numberOfClasses)
knnUniform.fit(X_train, y_train)

misclassifiedNum = 1
predictions, neighbors = knnUniform.predict(X_test)
print("3 misclassified examples in  model no 8:  k=7, non-weighted KNN, no feature normalization:\n ")

for i in range(2000):
    if predictions[i] != y_test[i]:
        print("misclassified sample " + str(misclassifiedNum) +" :")
        print("  Predicted label: " + str(predictions[i]) + "(" + personality_types[predictions[i]] + ")")
        print("  Actual label:    " + str(y_test[i]) + "(" + personality_types[y_test[i]] + ")")
        print("  nearest Neighbours:  ", end =" ")
        for neighbour in neighbors[i]:
             print(str(y_train[neighbour]) + "(" + personality_types[y_train[neighbour]] + ")", end =" ")
        print("\n")
        misclassifiedNum += 1

    if misclassifiedNum > 3:
        break
