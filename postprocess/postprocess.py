from sklearn import tree
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import graphviz

# https://scikit-learn.org/stable/modules/tree.html#tree
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier


df = pd.read_csv("bbmat.csv")
# print(df)

arr = df[["10pctl"]].to_numpy()[:,0]


res = np.convolve(arr, [1, 1, 1, 0, -1, -1, -1], 'valid')
# print(peaks)

# find peaks, prominence must be at least 99th percentile of res
pct = np.percentile(res, 99)
peaks, properties = find_peaks(res, prominence=pct)
print(peaks)
# print(properties["prominences"])

# peaks = np.append(peaks, len(arr))
# print(peaks)
# plt.plot(res)
# plt.show()

# generate class labels (each peak starts a new class)
X = np.zeros(arr.shape)
for i in peaks:
    X[i:] += 1

# generate class labels before vs after first peak
# X = np.zeros(arr.shape)
# X[:peaks[0]] = -1
# X[peaks[0]:] = 1


# data is currently a sequence
# will be converted into a feature vector, where each vector entry says whether 
# one feature appears before another in the sequence

# extract one sequence per row
seqs = df[df.columns[8:]]

# get complete alphabet
alphabet = {}

def add_to_alphabet(s):
    # print(s)
    if s in alphabet:
        return
    else:
        if type(s) == str:
            alphabet[s] = len(alphabet)
    

seqs.applymap(add_to_alphabet)
print(alphabet)

Y = np.zeros((arr.shape[0], len(alphabet)**2))

# generate feature names
feature_names = ["" for i in range(len(alphabet)**2)]

for i,si in alphabet.items():
    for j,sj in alphabet.items():
        feature_names[si * len(alphabet) + sj] = i + " < " + j


for row in seqs.iterrows():
    r = row[0]
    l = row[1].to_list()

    # all pairwise i before j
    for i in range(len(l)):
        for j in range(i+1, len(l)):

            if type(l[i]) == str and type(l[j]) == str:

                # compute feature corresponding to i appears before j
                si = alphabet[l[i]]
                sj = alphabet[l[j]]
                feat = si * len(alphabet) + sj
                # print(r, feat)
                Y[r, feat] = 1

clf = tree.DecisionTreeClassifier(max_depth=4
# , class_weight="balanced"
#, max_leaf_nodes=10
)
# named X and Y backwards above
clf = clf.fit(Y, X)
dot_data = tree.export_graphviz(clf, out_file=None, filled=True
,feature_names=feature_names
)
graph = graphviz.Source(dot_data)
graph.render("graph")