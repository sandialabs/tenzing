import sys
import json
import math

from sklearn import tree
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import graphviz

# https://scikit-learn.org/stable/modules/tree.html#tree
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

"""
FIXME: right now all operations are not distinguishable.
There could be two streamsyncs of the same stream, and the same operation will appear twice in the
order.
"""


csvPath = sys.argv[1]

# no header row
# index|1st pct|10th|50th|90th|99th|sequence json
df = pd.read_csv(csvPath, delimiter='|', header=None)
print(df)

# sort rows by 10th pctl time
df = df.sort_values(by=2)

# arr = df[["10pctl"]].to_numpy()[:,0]
arr = df.iloc[:,2].to_numpy()
print(arr[:10])



# convolution with step function to help find peaks
# keep only valid parts of the convolution, which means the 0th
# result index is the kernel centered on index len/2 of the data
cKernel = [1, 1, 1, 0, -1, -1, -1]
res = np.convolve(arr, [1, 1, 1, 0, -1, -1, -1], 'valid')
cOffset = len(cKernel) // 2


# find peaks, prominence must be at least 99th percentile of res
# the peak position is the first index after the jump up
pct = np.percentile(res, 90)
print(pct)
peaks, properties = find_peaks(res, prominence=pct)
peaks += cOffset
print(peaks)
print(properties["prominences"])
print(arr[peaks])



# peaks = np.append(peaks, len(arr))
# print(peaks)
# plt.plot(res)
# plt.show()



# generate class labels (each peak is the beginning of a new class)
# Y = np.zeros(arr.shape)
# for i in peaks:
#     Y[i:] += 1


# generate class labels before vs after first peak
Y = np.zeros(arr.shape)
Y[:peaks[1]] = -1
Y[peaks[1]:] = 1

print(Y[peaks])

# data is currently a sequence
# will be converted into a feature vector, where each vector entry says whether 
# one feature appears before another in the sequence

# extract one sequence per row
seqs = df[df.columns[7:]]

print(seqs)



# figure out which streams are present
streams = set()
for row in seqs.iterrows():
    r = row[0]
    l = row[1].to_list()

    for e in l:
        if type(e) != str and math.isnan(e):
            continue
        try:
            j = json.loads(e)
            try:
                streams.add(j["stream"])
            except KeyError:
                continue
        except TypeError as err:
            print(e, err)
            sys.exit(1)

print(streams)


# figure out which operations are present
nonSyncStreamOps = set()
for row in seqs.iterrows():
    r = row[0]
    l = row[1].to_list()

    for e in l:
        if type(e) != str and math.isnan(e):
            continue
        try:
            j = json.loads(e)
        except TypeError as err:
            continue

        if "stream" in j.keys():
            nonSyncStreamOps.add(j["name"])
nonSyncStreamOps.remove('CudaStreamWaitEvent-anon')
nonSyncStreamOps.remove('CudaEventRecord-anon')


print(nonSyncStreamOps)

# generate feature vectors where each feature is whether two symbols are in the same stream
X_sameStream = np.zeros((arr.shape[0], len(nonSyncStreamOps)**2))

ri = 0
for row in seqs.iterrows():
    r = row[0]
    l = row[1].to_list()

    # get stream of each symbol
    streams = {}

    for e in l:
        if type(e) != str and math.isnan(e):
            continue
        try:
            j = json.loads(e)
        except TypeError as err:
            continue
        if "stream" in j.keys():
            streams[j["name"]] = j["stream"]

    for i, si in enumerate(nonSyncStreamOps):
        for j, sj in enumerate(nonSyncStreamOps):
            if streams[si] == streams[sj]:
                X_sameStream[ri, i * len(nonSyncStreamOps) + j] = 1

    ri += 1

X_sameStream_names = []
for i, si in enumerate(nonSyncStreamOps):
    for j, sj in enumerate(nonSyncStreamOps):
        X_sameStream_names += [si + " and " + sj]

X = np.hstack((X_sameStream, ))
feature_names = X_sameStream_names
print(X.shape)


# remove any features that are identical for the whole dataset
# prefer to keep earlier features
i = 0
while i < X.shape[1]:
    j = i + 1
    while  j < X.shape[1]:
        if np.all(X[:, i] == X[:, j]):
            print("features", i, j, "are identical")
            X = np.delete(X, j, 1)
            feature_names = feature_names[:j] + feature_names[j+1:]
            j -= 1
            
        j += 1
    i += 1

print(feature_names)


# train decition tree
clf = tree.DecisionTreeClassifier(max_depth=6
,criterion="entropy"
# ,class_weight="balanced"
,max_leaf_nodes=20
)

clf = clf.fit(X, Y)
tree.export_graphviz(clf, out_file="trained.dot", filled=True
,feature_names=feature_names
)
# dot_data = tree.export_graphviz(clf, out_file=None, filled=True
# ,feature_names=feature_names
# )
# graph = graphviz.Source(dot_data)
# graph.render("graph")


sys.exit(1)

nonSyncNames = {}

# get complete alphabet
alphabet = {}
gpu_alphabet = {} # gpu operations only


gpu_prefixes = ["yr(", "yl(","y(","Scatter("]

# here we assume each sync operation has a unique name, even though
# that's not true right now
def parse_op(s):
    """turn op string `s` into a symbol,stream tuple
    NaN is None, None
    Start is Start, None
    End is End, None (FIXME?)
    """
    if type(s) == str:
        ss = s.strip()
        for i, p in enumerate(gpu_prefixes):
            if p in ss:
                return ss[:len(p)-1], int(ss[len(p):-1])
        if "StreamSync" in ss:
            return "StreamSync", None
        if "StreamWait" in ss:
            return "StreamWait", None
        return ss, None
    return None, None



def add_to_alphabet(op):

    symbol, stream = parse_op(op)

    if symbol:
        if symbol in alphabet:
            pass
        else:
            alphabet[symbol] = len(alphabet)

    if stream:
        if symbol in gpu_alphabet:
            pass
        else:
            gpu_alphabet[symbol] = len(gpu_alphabet)     
    

seqs.applymap(add_to_alphabet)
print(alphabet)
print(gpu_alphabet)

# generate feature vectors where each feature is whether one symbol comes before another
# in the sequence
X_ordering = np.zeros((arr.shape[0], len(alphabet)**2))

# generate feature names
ordering_feature_names = ["" for i in range(len(alphabet)**2)]
for i,si in alphabet.items():
    for j,sj in alphabet.items():
        ordering_feature_names[si * len(alphabet) + sj] = i + " < " + j


for row in seqs.iterrows():
    r = row[0]
    l = row[1].to_list()

    # all pairwise i before j
    for i in range(len(l)):
        for j in range(i+1, len(l)):

            symi, stri = parse_op(l[i])
            symj, strj = parse_op(l[j])

            if symi is not None and symj is not None:
                # compute feature corresponding to i appears before j
                si = alphabet[symi]
                sj = alphabet[symj]
                feat = si * len(alphabet) + sj
                # print(r, feat)
                X_ordering[r, feat] = 1




## generate features for which GPU operations occur in the same stream
X_co = np.zeros((arr.shape[0], len(gpu_alphabet)**2))



def fill_co2(r, row):
    l = row.to_list()
    for i, li in enumerate(l):
        for j, lj in enumerate(l[i+1:]):
            symi, stri = parse_op(li)
            symj, strj = parse_op(lj)
            if stri is not None and strj is not None:
                if stri == strj: # same stream
                    X_co[r, gpu_alphabet[symi] * len(gpu_alphabet) + gpu_alphabet[symj]] = 1

for row in seqs.iterrows():
    r = row[0]
    fill_co2(r, row[1])

co_feature_names = ["" for i in range(len(gpu_alphabet.keys())**2)]
for i,si in enumerate(gpu_alphabet.keys()):
    for j,sj in enumerate(gpu_alphabet.keys()):
        co_feature_names[i * len(gpu_alphabet.keys()) + j] = si + "&" + sj

"""
## generate features for which GPU operations co-occur
X_co = np.zeros((arr.shape[0], len(gpu_alphabet)**2))

# needle, haystack
def stream_for(n, h):
    if n in h:
        h = h[len(n):-1]
        return int(h)
    return None

def fill_co(r, row):
    l = row.to_list()
    for i, pi in enumerate(gpu_alphabet):
        for j, pj in enumerate(gpu_alphabet):
            if pi in l and pj in l:
                X_co[r, i * len(gpu_alphabet) + j] = 1

for row in seqs.iterrows():
    r = row[0]
    fill_co(r, row[1])

co_feature_names = ["" for i in range(len(gpu_alphabet)**2)]
for i,si in gpu_alphabet.items():
    for j,sj in gpu_alphabet.items():
        co_feature_names[si * len(gpu_alphabet) + sj] = i + "&" + j
"""

X = np.hstack((X_co, X_ordering))
print(X.shape)
feature_names = co_feature_names + ordering_feature_names

# remove any features that are identical for the whole dataset
# prefer to keep earlier features
i = 0
while i < X.shape[1]:
    j = i + 1
    while  j < X.shape[1]:
        if np.all(X[:, i] == X[:, j]):
            print(i, j)
            X = np.delete(X, j, 1)
            feature_names = feature_names[:j] + feature_names[j+1:]
            j -= 1
            
        j += 1
    i += 1


clf = tree.DecisionTreeClassifier(max_depth=6
,criterion="entropy"
# ,class_weight="balanced"
,max_leaf_nodes=20
)

clf = clf.fit(X, Y)
dot_data = tree.export_graphviz(clf, out_file=None, filled=True
,feature_names=feature_names
)
graph = graphviz.Source(dot_data)
graph.render("graph")