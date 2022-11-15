
"""
BBM409 Introduction to Machine Learning Lab. Fall 2022.
Assignment 1: PART 2 : Energy Efficiency Estimation (Regression)

Contributors:
Ali Argun Sayilgan : 21827775
Mehmet Giray Nacakci : 21989009
"""

import pandas as pd
import numpy as np
import random
import time
from IPython.display import display
pd.set_option('display.max_columns', None)

# import dataset
df = pd.read_csv("../energy_efficiency_data.csv", encoding='cp1252')
df.describe()
X = df.drop(['Heating_Load', 'Cooling_Load'], axis=1).to_numpy()
Y = df[['Heating_Load', 'Cooling_Load']].to_numpy()


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


def StandardScaler():
    def __init__(self):
        None

    def fit_transform(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std


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


class KNeighborsRegressor():
    def __init__(self, n_neighbors, weights='uniform'):
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
            sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:self.n_neighbors]

            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]
            neigh_ind.append(ind_list)
            dist.append(dist_list)

        if return_distance:
            return np.array(dist), np.array(neigh_ind)

        return np.array(neigh_ind)

    def predict(self, X_test):

        if self.weights == 'uniform':
            neighbor_indices_of_all_rows = self.kneighbors(X_test)

            y_pred = []
            for row in neighbor_indices_of_all_rows:
                neighbors_y_sum_heating = 0
                neighbors_y_sum_cooling = 0

                # predicted value is the average value of neighbors
                for neighbor_index in row:
                    neighbors_y_sum_heating += self.y_train[neighbor_index][0]
                    neighbors_y_sum_cooling += self.y_train[neighbor_index][1]
                y_pred.append([neighbors_y_sum_heating / len(row), neighbors_y_sum_cooling / len(row)])

            return np.array(y_pred)


        # Weighted KNN
        elif self.weights == 'distance':

            neighbor_distances_of_all_rows, neighbor_indices_of_all_rows = self.kneighbors(X_test, return_distance=True)
            inverse_distances_of_all_rows = 1 / neighbor_distances_of_all_rows
            y_pred = []

            for i, row in enumerate(inverse_distances_of_all_rows):
                neighbors_weighted_y_sum_heating = 0
                neighbors_weighted_y_sum_cooling = 0

                # predicted value is the weighted average value of neighbors
                for j, inverse_distance in enumerate(row):
                    neighbors_weighted_y_sum_heating += inverse_distance * \
                                                        self.y_train[neighbor_indices_of_all_rows[i][j]][0]
                    neighbors_weighted_y_sum_cooling += inverse_distance * \
                                                        self.y_train[neighbor_indices_of_all_rows[i][j]][1]

                y_pred.append(
                    [neighbors_weighted_y_sum_heating / np.sum(row), neighbors_weighted_y_sum_cooling / np.sum(row)])

            return np.array(y_pred)


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


# classification metric: mean absolute error
def MAE(pred, actual, multi_output=None):
    mae = np.sum(np.absolute((pred.astype("float") - actual.astype("float"))))
    mae /= float(pred.shape[0] * pred.shape[1])
    if multi_output is None:
        return np.average(mae)
    return mae


def cross_val_score(X, Y, cv, pipeline):
    mae_heating_folds = []
    mae_cooling_folds = []

    # for each Fold of 5-fold-validation
    for (x_train, x_test, y_train, y_test) in cv.split(X,Y):
        y_pred = pipeline.execute(x_train, x_test, y_train)

        mae_heating_folds.append(MAE(y_pred[:, [0]], y_test[:, [0]]))
        mae_cooling_folds.append(MAE(y_pred[:, [1]], y_test[:, [1]]))

    # averages of folds
    mae_heating_folds.append(sum(mae_heating_folds)/5)
    mae_cooling_folds.append(sum(mae_cooling_folds)/5)

    return mae_heating_folds, mae_cooling_folds





""" Run non-weighted and weighted KNN models   (20 variations) """

cv = KFold(5, shuffle=True, random_state=24)
scaler = MinMaxScaler()
neighborVariations = [1,3,5,7,9]

heatingTable_columns = []
coolingTable_columns = []


def run_all_models():
    print(" \nResults of 20 KNN model variations will be ready in less than  30 SECONDS.  Please wait... \n")
    start = time.time()


    """  ***   NON-WEIGHTED KNN   *** """
    for k in neighborVariations:  # THIS LOOP TAKES LESS THAN 15 SECONDS TO COMPLETE
        knnUniform = KNeighborsRegressor(n_neighbors=k, weights='uniform')

        # with feature normalization
        pipeline = Pipeline(scaler=scaler, classifier=knnUniform)
        heating_MAEs, cooling_MAEs = cross_val_score(X, Y, cv, pipeline)
        heatingTable_columns.append(heating_MAEs)
        coolingTable_columns.append(cooling_MAEs)

        # without feature normalization
        pipeline = Pipeline(classifier=knnUniform)
        heating_MAEs, cooling_MAEs = cross_val_score(X, Y, cv, pipeline)
        heatingTable_columns.append(heating_MAEs)
        coolingTable_columns.append(cooling_MAEs)


    """  ***   WEIGHTED KNN   *** """
    for k in neighborVariations:   # THIS LOOP TAKES LESS THAN 15 SECONDS TO COMPLETE
        knnDistance = KNeighborsRegressor(n_neighbors=k, weights='distance')

        # with feature normalization
        pipeline = Pipeline(scaler=scaler, classifier=knnDistance)
        heating_MAEs, cooling_MAEs = cross_val_score(X, Y, cv, pipeline)
        heatingTable_columns.append(heating_MAEs)
        coolingTable_columns.append(cooling_MAEs)

        # without feature normalization
        pipeline = Pipeline(classifier=knnDistance)
        heating_MAEs, cooling_MAEs = cross_val_score(X, Y, cv, pipeline)
        heatingTable_columns.append(heating_MAEs)
        coolingTable_columns.append(cooling_MAEs)


    # model calculations are finished.
    finish = time.time()
    seconds = finish-start
    print("Results of 20 KNN model variations are ready in the sections below. Thank you for your patience.")
    print('Elapsed time is:   %d   seconds\n' %seconds)

run_all_models()




""" Heating and Cooling  -  Cross Validation Scores """

print("\n------ HEATING LOAD - Mean Absolute Errors - for 20 model variations ------\n")
heating_rows = np.transpose(np.array(heatingTable_columns))
heating_table = pd.DataFrame(heating_rows, columns = ['1: k=1 w- n+','2: k=1 w- n-','3: k=3 w- n+','4: k=3 w- n-','5: k=5 w- n+','6: k=5 w- n-','7: k=7 w- n+','8: k=7 w- n-','9: k=9 w- n+','10: k=9 w- n-','11: k=1 w+ n+','12: k=1 w+ n-','13: k=3 w+ n+','14: k=3 w+ n-','15: k=5 w+ n+','16: k=5 w+ n-','17: k=7 w+ n+','18: k=7 w+ n-','19: k=9 w+ n+','20: k=9 w+ n-'])
heating_table.index = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average of Folds']

display(heating_table.iloc[:, :10].head(6))
display(heating_table.iloc[:, 10:].head(6))
print("\nmodel variations encoding: \n k=  : k parameter of KNN \n w+  :       weighted KNN \n w-  :   non-weighted KNN \n n+  :    with feature normalization \n n-  : without feature normalization \n")


def draw_cooling_table():
    print("\n------ COOLING LOAD - Mean Absolute Errors - for 20 model variations ------\n")
    cooling_rows = np.transpose(np.array(coolingTable_columns))
    cooling_table = pd.DataFrame(cooling_rows, columns = ['1: k=1 w- n+','2: k=1 w- n-','3: k=3 w- n+','4: k=3 w- n-','5: k=5 w- n+','6: k=5 w- n-','7: k=7 w- n+','8: k=7 w- n-','9: k=9 w- n+','10: k=9 w- n-','11: k=1 w+ n+','12: k=1 w+ n-','13: k=3 w+ n+','14: k=3 w+ n-','15: k=5 w+ n+','16: k=5 w+ n-','17: k=7 w+ n+','18: k=7 w+ n-','19: k=9 w+ n+','20: k=9 w+ n-'])
    cooling_table.index = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average of Folds']

    display(cooling_table.iloc[:, :10].head(6))
    display(cooling_table.iloc[:, 10:].head(6))

draw_cooling_table()
print("\nmodel variations encoding: \n k=  : k parameter of KNN \n w+  :       weighted KNN \n w-  :   non-weighted KNN \n n+  :    with feature normalization \n n-  : without feature normalization \n")




""" Testing feature Standardization on Cooling_Load """

scaler = StandardScaler()
coolingTable_columns = []

run_all_models()
draw_cooling_table()

print("\nmodel variations encoding: \n k=  : k parameter of KNN \n w+  :       weighted KNN \n w-  :   non-weighted KNN \n n+  :  Scaling with feature  STANDARDIZATION \n n-  :  no Scaling \n")

