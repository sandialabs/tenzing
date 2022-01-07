import sys
import json
import math
import re

from sklearn import tree
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import graphviz
from io import StringIO

# https://scikit-learn.org/stable/modules/tree.html#tree
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier



csvPath = sys.argv[1]

# read csv in and ensure each line has the same number of delims
# since the first line may not have the most
with open(csvPath, "r") as f:
    lines = f.readlines()

    maxDelims = -1
    for line in lines:
        maxDelims = max(line.count('|'), maxDelims)

    for i, _ in enumerate(lines):
        delims = lines[i].count('|')
        lines[i] = lines[i].strip() + '|' * (maxDelims - delims) + '\n'
    csvStr = ''.join(lines)


# no header row
# index|1st pct|10th|50th|90th|99th|sequence json
df = pd.read_csv(StringIO(csvStr), delimiter='|', header=None)
print(df)


# sort rows by 10th pctl time
df = df.sort_values(by=2)

# arr = df[["10pctl"]].to_numpy()[:,0]
arr = df.iloc[:,2].to_numpy()
print(arr[:10])



# convolution with step function to help find peaks
# keep only valid parts of the convolution, which means the 0th
# result index is the kernel centered on index len/2 of the data
krf = 0.005# radius fraction
kr = int(math.ceil(len(arr) * krf))
cKernel = [1] * kr + [0] + [-1] * kr
res = np.convolve(arr, cKernel, 'valid')
cOffset = len(cKernel) // 2


# find peaks, prominence must be at least 99th percentile of res
# the peak position is the first index after the jump up
pct = 90
pct = 99
cutoff = np.percentile(res, pct)
print(cutoff)
peaks, properties = find_peaks(res, prominence=cutoff)
peaks += cOffset
print("peaks:", peaks)
print("prominences", properties["prominences"])
print(arr[peaks])



# peaks = np.append(peaks, len(arr))
# print(peaks)
# plt.plot(res)
# plt.plot(arr)
# plt.axhline(y=cutoff, color='r', linestyle='-')

fig, axs = plt.subplots(3, sharex=True, figsize=(5,5))

axs[0].plot(arr, color='black')
axs[0].set_ylabel("Elapsed Time (s)")

axs[1].axhline(y=cutoff, color='gray', linestyle='-', label=f"{pct}th percentile threshold")
axs[1].plot(res, color='black')
axs[1].set_ylabel("Convolution Result")
axs[1].legend()

axs[2].set_ylabel("Class Boundaries")
axs[2].plot(arr, color='gray')
axs[2].axes.yaxis.set_ticks([])
for peak in peaks:
    axs[2].axvline(x=peak, color='r', linestyle='-')

axs[2].set_xlabel("Implementation")


fig.align_labels()
fig.tight_layout()

fig.subplots_adjust(left=0.20) # relative position of left edge of subplots
fig.text(0, 0.85, "(a)")
fig.text(0, 0.55, "(b)")
fig.text(0, 0.24, "(c)")

plt.savefig("classes.pdf")
plt.clf()



# generate class labels (each peak is the beginning of a new class)
Y = np.zeros(arr.shape, dtype=int)
for i in peaks:
    Y[i:] += 1


# generate class labels before vs after first peak
# Y = np.zeros(arr.shape, dtype=int)
# Y[:peaks[1]] = 0
# Y[peaks[1]:] = 1

nClasses = len(np.unique(Y))
print("peak positions:", Y[peaks])

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

print("found streams:", streams)

def is_sync_op(d):
    if d.get("kind") == "CudaEventRecord":
        return True
    if d.get("kind") == "CudaEventSync":
        return True
    if d.get("kind") == "CudaStreamWaitEvent":
        return True
    return False

# figure out which operations are present
nonSyncStreamOps = set()
nonSyncOps = set()
allOps = set()
for row in seqs.iterrows():
    i = row[0]
    l = row[1].to_list() # list of json strings
    l = filter(lambda e: type(e) == str, l) # remove non-string
    l = map(json.loads, l) # convert to json
    l = list(l)

    for j in l:
        if "stream" in j.keys() and not is_sync_op(j):
            nonSyncStreamOps.add(j["name"])
        if not is_sync_op(j):
            nonSyncOps.add(j["name"])
        allOps.add(j["name"])



def remove_if_present(d, key):
    """ call d.remove(key) and supress KeyError """
    try:
        d.remove(key)
    except KeyError:
        pass

print(nonSyncOps)
print(nonSyncStreamOps)
print(allOps)

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

# generate the name for this feature
X_sameStream_names = []
for i, si in enumerate(nonSyncStreamOps):
    for j, sj in enumerate(nonSyncStreamOps):
        X_sameStream_names += [si + " and " + sj]

def list_index_all(l, e):
    if [] == l:
        return []
    else:
        try:
            i = l.index(e)
        except ValueError:
            return []
        return [i] + list_index_all(l[i+1:], e)

def any_lt(xs, ys):
    for x in xs:
        for y in ys:
            if x < y:
                return True
    return False

# generate feature vector for operator ordering
X_order = np.zeros((arr.shape[0], len(allOps)**2))
X_order_names = []
for i, si in enumerate(allOps):
    for j, sj in enumerate(allOps):
        X_order_names += [si + " before " + sj]


for ri, row in enumerate(seqs.iterrows()):
    r = row[0]
    l = row[1].to_list()
    l = filter(lambda e: type(e) == str, l) # keep strings
    l = map(json.loads, l) # covert to json
    # l = filter(lambda e: not is_sync_op(e), l) # keep non-sync ops
    l = list(l)
    filtered = list(map(lambda d: d["name"], l)) # list of names of non-sync ops

    for i, si in enumerate(allOps):
        for j, sj in enumerate(allOps):
            ijs = list_index_all(filtered, sj)
            iis = list_index_all(filtered, si)

            # if any ii before any ij
            if any_lt(iis, ijs):
                X_order[ri, i * len(allOps) + j] = 1


# combine all features
X = np.hstack((X_sameStream, X_order))
feature_names = [] + X_sameStream_names + X_order_names
print(X.shape)

# remove any features that have the same value for the whole dataset
i = 0
while i < X.shape[1]:
    if np.all(X[:, i] == X[0, i]):
        print("feature", i, "same for whole dataset", feature_names[i])
        X = np.delete(X, i, 1)
        feature_names = feature_names[:i] + feature_names[i+1:]
        i -= 1
    i += 1

# remove any features that are identical for the whole dataset
# prefer to keep earlier features
i = 0
while i < X.shape[1]:
    j = i + 1
    while  j < X.shape[1]:
        if np.all(X[:, i] == X[:, j]):
            print("features", i, "and", j, "are identical", feature_names[i], feature_names[j])
            X = np.delete(X, j, 1)
            feature_names = feature_names[:j] + feature_names[j+1:]
            j -= 1
            
        j += 1
    i += 1

print(feature_names)


def rewrite_streams_label(mo):
    return f'label="{mo.group(1)}, {mo.group(2)} different streams'

def rewrite_order_label(mo):
    return f'label="{mo.group(2)} before {mo.group(1)}'

maxDepth = math.log2(X.shape[0] / 5)

print("max_depth", maxDepth)


maxLeafNodesRange = (nClasses, 9*nClasses)
maxDepthRange = (1, 9)

def get_f1(X, Y, maxLeafNodes):

    maxDepth = maxLeafNodes-1
    
    #train tree
    clf = tree.DecisionTreeClassifier(max_depth=maxDepth
    ,criterion="entropy"
    ,class_weight="balanced"
    ,max_leaf_nodes=maxLeafNodes
    ,min_impurity_decrease=0.001 # avoid splitting good nodes
    )
    clf = clf.fit(X, Y)

    # confusion matrix
    cm = np.zeros((nClasses, nClasses))
    y = clf.predict(X).astype(int)
    for s in range(X.shape[0]):
        i,j = Y[s], y[s]
        cm[i,j] += 1

    tp = np.zeros(nClasses, dtype=int)
    fp = np.zeros(nClasses, dtype=int)
    tn = np.zeros(nClasses, dtype=int)
    fn = np.zeros(nClasses, dtype=int)
    for c in range(nClasses):
        # binary classification for class c
        # i: true label
        # j: predicted label
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i == c and j == c:
                    tp[c] += cm[i,j]
                if i != c and j == c: # true is neg, pred is positive
                    fp[c] += cm[i,j]
                if i != c and j != c:
                    tn[c] += cm[i,j]
                if i == c and j != c: # ture is positive, pred is negative
                    fn[c] += cm[i,j]

    # micro: weight all items equally
    # macro: weight all classes equally
    recall_u = np.sum(tp) / (np.sum(tp) + np.sum(fn))
    precision_u = np.sum(tp) / (np.sum(tp) + np.sum(fp))
    # print("recall_u:   ", recall_u)
    # print("precision_u:", precision_u)
    f1_u = 2*recall_u*precision_u/(recall_u+precision_u)
    # print("f1_u:       ", f1_u)
    # recall_m = np.sum(tp / (tp+fn)) / nClasses
    # precision_m = np.sum(tp / (tp+fp)) / nClasses
    # print("recall_m:   ", recall_m)
    # print("precision_m:", precision_m)
    # print("f1_m:       ", 2*recall_m*precision_m/(recall_m+precision_m))
    # print("avg acc:    ", np.sum((tp+tn)/(tp+tn+fp+fn))/nClasses)
    # print("avg err:    ", np.sum((fp+fn)/(tp+tn+fp+fn))/nClasses)
    return f1_u, clf

def dump_dot(clf, path):   
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True
    ,feature_names=feature_names
    )
    lines = dot_data.splitlines()
    for i in range(len(lines)):
        lines[i] = re.sub('label="(.*?) and (.*?) <= 0.5', rewrite_streams_label, lines[i])
        lines[i] = re.sub('label="(.*?) before (.*?) <= 0.5', rewrite_order_label, lines[i])
    dot_data = '\n'.join(lines)
    with open(path, "w") as f:
        f.write(dot_data)


mln = nClasses
f1, clf = get_f1(X, Y, mln)
f1s = []
mlns = []
while True:

    print(f"max_leaf_nodes: {mln}")
    print(f"f1: {f1}")

    dump_dot(clf, f"dt_{mln}.dot")

    f1s += [f1]
    mlns += [mln]

    nf1, clf = get_f1(X, Y, mln+1)
    if nf1 > f1:
        mln = mln+1
        f1 = nf1
        continue

    nf1, clf = get_f1(X, Y, mln+2)
    if nf1 > f1:
        mln = mln+2
        f1 = nf1
        continue

    nf1, clf = get_f1(X, Y, mln+3)
    if nf1 > f1:
        mln = mln+3
        f1 = nf1
        continue
    
    break

print(f"max_leaf_nodes: {mln}")

plt.figure(figsize=(4,3))
plt.plot(mlns, f1s)
plt.xlabel("# Leaf Nodes")
plt.ylabel("f1-micro score")
plt.tight_layout()
plt.savefig("f1.pdf")
plt.clf()

sys.exit(1)

dotfilePath = f"dt_{mln}_{md}.dot"

# train decision tree
clf = tree.DecisionTreeClassifier(max_depth=md
,criterion="entropy"
,class_weight="balanced"
,max_leaf_nodes=mln
,min_impurity_decrease=0.001 # avoid splitting good nodes
)

clf = clf.fit(X, Y)
tree.export_graphviz(clf, out_file=dotfilePath, filled=True
,feature_names=feature_names
)
dot_data = tree.export_graphviz(clf, out_file=None, filled=True
,feature_names=feature_names
)


lines = dot_data.splitlines()

for i in range(len(lines)):
    lines[i] = re.sub('label="(.*?) and (.*?) <= 0.5', rewrite_streams_label, lines[i])
    lines[i] = re.sub('label="(.*?) before (.*?) <= 0.5', rewrite_order_label, lines[i])


dot_data = '\n'.join(lines)
# print(dot_data)

with open(dotfilePath, "w") as f:
    f.write(dot_data)

# graph = graphviz.Source(dot_data)
# graph.render("graph")

# confusion matrix

cm = np.zeros((nClasses, nClasses))

y = clf.predict(X).astype(int)
# print(y, Y)
for s in range(X.shape[0]):
    i,j = Y[s], y[s]
    cm[i,j] += 1

# print(cm)



tp = np.zeros(nClasses, dtype=int)
fp = np.zeros(nClasses, dtype=int)
tn = np.zeros(nClasses, dtype=int)
fn = np.zeros(nClasses, dtype=int)
for c in range(nClasses):
    # binary classification for class c
    # i: true label
    # j: predicted label
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == c and j == c:
                tp[c] += cm[i,j]
            if i != c and j == c: # true is neg, pred is positive
                fp[c] += cm[i,j]
            if i != c and j != c:
                tn[c] += cm[i,j]
            if i == c and j != c: # ture is positive, pred is negative
                fn[c] += cm[i,j]

# micro: weight all items equally
# macro: weight all classes equally
recall_u = np.sum(tp) / (np.sum(tp) + np.sum(fn))
precision_u = np.sum(tp) / (np.sum(tp) + np.sum(fp))
print("recall_u:   ", recall_u)
print("precision_u:", precision_u)
f1_u = 2*recall_u*precision_u/(recall_u+precision_u)
print("f1_u:       ", f1_u)
recall_m = np.sum(tp / (tp+fn)) / nClasses
precision_m = np.sum(tp / (tp+fp)) / nClasses
print("recall_m:   ", recall_m)
print("precision_m:", precision_m)
print("f1_m:       ", 2*recall_m*precision_m/(recall_m+precision_m))
print("avg acc:    ", np.sum((tp+tn)/(tp+tn+fp+fn))/nClasses)
print("avg err:    ", np.sum((fp+fn)/(tp+tn+fp+fn))/nClasses)

f1s[md, mln] = f1_u

# rules for classes

plt.imshow(f1s)
# plt.plot(f1_x, f1_y)
plt.xlabel("# Leaf Nodes")
plt.ylabel("Max Depth")
plt.savefig("f1.pdf")
plt.clf()

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