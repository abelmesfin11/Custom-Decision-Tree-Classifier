#!/usr/bin/env python

import numpy as np
import pandas as pd
from data_loader import load_simulated_data, load_transplant_data
from sklearn.model_selection import train_test_split


def accuracy(Y, Yhat):
    """
    Function for computing accuracy

    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    return np.sum(Y == Yhat) / len(Y)

def entropy(data, outcome_name):
    """
    Compute entropy of assuming the outcome is binary

    data: a pandas dataframe
    outcome_name: a string corresponding to name of the outcome variable
    """
    counts = data[outcome_name].value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts + 1e-9))  # To avoid log(0), add small num


def weighted_entropy(data1, data2, outcome_name):
    """
    Calculate the weighted entropy of two datasets

    data1: a pandas dataframe
    data2: a pandas dataframe
    outcome_name: a string corresponding to name of the outcome varibale
    """
    total_len = len(data1) + len(data2)
    return (len(data1) * entropy(data1, outcome_name) + len(data2) * entropy(data2, outcome_name)) / total_len


class Vertex:
    """
    Class for defining a vertex in a decision tree
    """

    def __init__(self, feature_name=None, threshold=None, prediction=None):

        self.left_child = None
        self.right_child = None
        self.feature_name = feature_name # name of feature to split on
        self.threshold = threshold # threshold of feature to split on
        self.prediction = prediction # predicted value -- applies only to leaf nodes


class DecisionTree:
    """
    Class for building decision trees
    """

    def __init__(self, max_depth=np.inf):

        self.max_depth = max_depth
        self.root = None

    def _get_best_split(self, data, outcome_name):
        """
        Method to compute the best split of the data to minimize entropy

        data: pandas dataframe
        outcome_name: a string corresponding to name of the outcome variable

        Returns
        ------
        A tuple consisting of:
        (i) String corresponding to name of the best feature
        (ii) Float corresponding to value to split the feature on
        (iii) pandas dataframe consisting of subset of rows of data where best_feature < best_threshold
        (iv) pandas dataframe consisting of subset of rows of data where best_feature >= best_threshold
        """

        best_entropy = entropy(data, outcome_name)
        best_feature = None
        best_threshold = 0
        data_left, data_right = None, None

        # All features without the outcome_name
        features = [col for col in data.columns if col != outcome_name]

        for feature in features:

            # All unique values of a column as threshold
            unique_values = data[feature].unique()

            for threshold in unique_values:
                left_split = data[data[feature] < threshold]
                right_split = data[data[feature] >= threshold]

                # Useless split
                if len(left_split) == 0 or len(right_split) == 0:
                    continue

                # Weighted entropy of a split
                split_entropy = weighted_entropy(left_split, right_split, outcome_name)
                if split_entropy < best_entropy:
                    # Update
                    best_entropy = split_entropy
                    best_feature = feature
                    best_threshold = threshold
                    data_left = left_split
                    data_right = right_split

        return best_feature, best_threshold, data_left, data_right

    def _build_tree(self, data, outcome_name, curr_depth=0):
        """
        Recursive function to build a decision tree. Refer to the HW pdf
        for more details on the implementation of this function.

        data: pandas dataframe
        outcome_name: a string corresponding to name of the outcome varibale
        curr_depth: integer corresponding to current depth of the tree
        """
        # Base cases
        if len(data[outcome_name].unique()) == 1 or curr_depth >= self.max_depth:
            return Vertex(prediction=data[outcome_name].mode()[0])

        best_feature, best_threshold, left_split, right_split = self._get_best_split(data, outcome_name)

        # If no valid split found, create a leaf node
        if best_feature is None:
            return Vertex(prediction=data[outcome_name].mode()[0])

        # Create a decision node with the best feature and threshold
        node = Vertex(best_feature, best_threshold)

        # Recursively build the left and right subtrees by splitting data
        node.left_child = self._build_tree(left_split, outcome_name, curr_depth + 1)
        node.right_child = self._build_tree(right_split, outcome_name, curr_depth + 1)

        return node

    def fit(self, Xmat, Y, outcome_name="Y"):
        """
        Fit a decision tree model using training data Xmat and Y.

        Xmat: pandas dataframe of features
        Y: numpy array of 0/1 outcomes
        outcome_name: string corresponding to name of outcome variable
        """

        data = Xmat.copy()
        data[outcome_name] = Y
        self.root = self._build_tree(data, outcome_name, 0)


    def _dfs_to_leaf(self, sample, node=None):
        """
        Perform a depth first traversal to find the leaf node that the given sample belongs to

        sample: dictionary mapping from feature names to values of the feature
        """
        # If no node is passed, begin at the root of the tree
        if node is None:
            node = self.root

        # If we've reached a leaf node, return its stored prediction
        if node.prediction is not None:
            return node.prediction

        # Go left or right in the tree based on the feature value
        if sample[node.feature_name] < node.threshold:
            return self._dfs_to_leaf(sample, node.left_child)
        else:
            return self._dfs_to_leaf(sample, node.right_child)


    def predict(self, Xmat):
        """
        Predict 0/1 labels for a data matrix

        Xmat: pandas dataframe
        """
        predictions = []
        for i in range(len(Xmat)):
            sample = {feature: Xmat[feature].iloc[i] for feature in Xmat.columns}
            prediction = self._dfs_to_leaf(sample)
            predictions.append(prediction)
        return np.array(predictions)

    def print_tree(self, vertex=None, indent="  "):
        """
        Function to produce text representation of the tree
        """

        # initialize to root node
        if not vertex:
            vertex = self.root

        # if we're at the leaf output the prediction
        if vertex.prediction is not None:
            print("Output", vertex.prediction)

        else:
            print(vertex.feature_name, "<", round(vertex.threshold, 2), "?")
            print(indent, "Left child: ", end="")
            self.print_tree(vertex.left_child, indent + indent)
            print(indent, "Right child: ", end="")
            self.print_tree(vertex.right_child, indent + indent)


def main():
    """
    Edit only the one line marked as # EDIT ME in this function. The rest is used for grading purposes
    """


    #################
    # Simulated data
    #################
    np.random.seed(333)
    Xmat, Y  = load_simulated_data()
    data = Xmat.copy()
    data["Y"] = Y

    # test for your predict method
    # by manually creating a decision tree
    model = DecisionTree()
    model.root = Vertex(feature_name="X2", threshold=1.2)
    model.root.left_child = Vertex(prediction=0)
    model.root.right_child = Vertex(feature_name="X1", threshold=1.2)
    model.root.right_child.left_child = Vertex(prediction=0)
    model.root.right_child.right_child = Vertex(prediction=1)
    print("-"*60 + "\n" + "Hand crafted tree for testing predict\n" + "-"*60)
    model.print_tree()
    Yhat = model.predict(Xmat)
    print("Accuracy of hand crafted tree", round(accuracy(Y, Yhat), 2), "\n")

    # test for your best split method
    print("-"*60 + "\n" + "Simple test for finding best split\n" + "-"*60)
    model = DecisionTree(max_depth=2)
    best_feature, threshold, _, _ = model._get_best_split(data, "Y")
    print("Best feature and threshold found", best_feature, round(threshold, 2), "\n")


    # test for your fit method
    model.fit(Xmat, Y)
    print("-"*60 + "\n" + "Algorithmically generated tree for testing build_tree\n" + "-"*60)
    model.print_tree()
    Yhat = model.predict(data)
    print("Accuracy of algorithmically generated tree", round(accuracy(Y, Yhat), 2), "\n")

    #####################
    # Transplant data
    #####################
    Xmat, Y = load_transplant_data()

    # create a train test split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xmat, Y, test_size=0.25, random_state=0)
    Xtrain.reset_index(inplace=True, drop=True)
    Xtest.reset_index(inplace=True, drop=True)

    # find best depth using a form of cross validation/bootstrapping
    possible_depths = [1, 2, 3, 4, 5]
    best_depth = 0
    best_accuracy = 0
    for depth in possible_depths:

        accuracies = []
        for i in range(5):
            Xtrain_i, Xval, Ytrain_i, Yval = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=i)
            Xtrain_i.reset_index(inplace=True, drop=True)
            Xval.reset_index(inplace=True, drop=True)
            model = DecisionTree(max_depth=depth)
            model.fit(Xtrain_i, Ytrain_i, "survival_status")
            accuracies.append(accuracy(Yval, model.predict(Xval)))

        mean_accuracy = sum(accuracies)/len(accuracies)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_depth = depth


    print("-"*60 + "\n" + "Hyperparameter tuning on transplant data\n" + "-"*60)
    print("Best depth =", best_depth, "\n")
    model = DecisionTree(max_depth=best_depth)
    model.fit(Xtrain, Ytrain, "survival_status")
    print("-"*60 + "\n" + "Final tree for transplant data\n" + "-"*60)
    model.print_tree()
    print("Test accuracy", round(accuracy(Ytest, model.predict(Xtest)), 2), "\n")


if __name__ == "__main__":
    main()
