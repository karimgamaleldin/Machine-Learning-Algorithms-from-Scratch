import numpy as np
from collections import Counter
import pandas as pd
import random


def euclidean_distance(a, b):
    # dist = sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) Slower version
    # dist = np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2)) faster version
    dist = np.linalg.norm(np.array(a) - np.array(b))  # already implemented in numpy
    return dist


def k_nearest_neighbors(data, predict, k=3):
    distances = []
    for group in data:
        for features in data[group]:
            euc_dist = euclidean_distance(features, predict)
            distances.append([euc_dist, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    # print(f"Confidence: {confidence}")
    return vote_result, confidence


df = pd.read_csv('dataset.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
full_data = df.astype(float).values.tolist() # To change string/int values to floats

# shuffling our data
random.shuffle(full_data) # shuffles inplace
test_size = 0.2
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

## Running our algorithm
correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote, conf = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
           print(f"Confidence of wrong vote: {conf}")
        total += 1

print(f"Accuracy: {correct/total}") # Our implementation is doing as well as that for scikit learn.


