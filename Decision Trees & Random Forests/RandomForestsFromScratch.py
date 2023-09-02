from DecisionTreesFromScratch import DecisionTree
import numpy as np
from collections import Counter
from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees =[]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            print(f"Creating tree {i + 1}")
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features, isRandomForest=True)

            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)


    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]


    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        preds = []
        for pred in tree_preds:
            counter = Counter(pred)
            most_common = counter.most_common(1)[0][0]
            preds.append(most_common)
        return np.array(preds)


data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  np.sum(y_test == predictions) / len(y_test)
print(acc)

