import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p
        self.labelled_points = []
        self.ids = (206968281, 206418642)
        self.labels = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        self.labels = np.unique(y)
        if len(self.labelled_points) != 0:
            # if we run the code again we need to make labelled points empty
            self.labelled_points = []
        i = 0
        for row in X:
            temp = []
            temp.append(row)
            temp.append(y[i])
            self.labelled_points.append(temp)
            i += 1



    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        predict_label = np.zeros(len(X), dtype=np.uint8)
        num_labels = len(self.labels)
        j = 0
        for row in X:
            neighbors = self.find_k_nearest_neighbors(row, self.labelled_points)
            histogram = [0]*num_labels
            for obj in neighbors:
                histogram[obj[0][1]] += 1
            max_neighbors_num = max(histogram)
            max_neighbors = []
            for i in range(len(histogram)):
                if histogram[i] == max_neighbors_num:
                    max_neighbors.append(i)
            if len(max_neighbors) == 1:
                predict_label[j] = max_neighbors[0]
                j += 1
            else:
                flag = True
                same_dist = []
                for neighbor in neighbors:
                    if (neighbor[0][1] in max_neighbors) and flag:
                        temp = []
                        temp.append(neighbor[0][1])
                        temp.append(neighbor[1])
                        same_dist.append(temp)
                        flag = False
                for neighbor1 in neighbors:
                    if(neighbor1[0][1] in max_neighbors) and (neighbor1[1] == same_dist[0][1]) and (neighbor1[0][1] != same_dist[0][0]):
                        temp = []
                        temp.append(neighbor1[0][1])
                        temp.append(neighbor1[1])
                        same_dist.append(temp)
                winning_label = min(same_dist, key=lambda a: a[0])
                true_label = winning_label[0]
                predict_label[j] = true_label
                j += 1
        return predict_label


    def minkowski_distance(self, x1: np.ndarray, x2: np.ndarray):
        """
        computing the minkowski distance of 2 given vectors
        :param x1: vector 1
        :param x2: vector 2
        :return: the distance
        """
        d = len(x1)
        sum = 0
        for i in range(d):
            sum += ((abs(x1[i] - x2[i])) ** self.p)
        return sum ** (1 / self.p)

    def find_k_nearest_neighbors(self, x: np.ndarray, train_set: np.ndarray):
        """
        finding the k nearest neighbors of a given vector from a training set
        :param x: the vector
        :param train_set: a given training set
        :return: the k nearest neighbors of x
        """
        distances = []

        for row in train_set:
            dist = self.minkowski_distance(x, row[0])
            temp =[]
            temp.append(row)
            temp.append(dist)
            distances.append(temp)

        sorted_dist = sorted(distances, key=lambda a: a[-1])
        return sorted_dist[:self.k]



def main():
    print("*" * 20)
    print("Started HW1_206968281_206418642.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)

if __name__ == "__main__":
    main()
