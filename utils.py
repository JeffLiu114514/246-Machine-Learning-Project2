from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

import pandas as pd

def load_newts(filepath, do_min_max=False):
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00528/dataset.csv', delimiter=';', header = 1)
    xvals_raw = data.drop(['ID', 'Green frogs', 'Brown frogs', 'Common toad', 'Tree frog', 'Common newt', 'Great crested newt', 'Fire-bellied toad'], axis=1)
    xvals = pd.get_dummies(xvals_raw, columns=['Motorway', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'MR', 'CR'])
    yvals = data['Fire-bellied toad']

    # optional min-max scaling
    if (do_min_max):
       for col in ['SR', 'NR', 'TR', 'VR', 'OR', 'RR', 'BR']:
           xvals_raw[col] = (xvals_raw[col] - xvals_raw[col].min())/(xvals_raw[col].max() - xvals_raw[col].min())
    xvals = pd.get_dummies(xvals_raw, columns=['Motorway', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'MR', 'CR'])
    return xvals.to_numpy(), yvals.to_numpy()

# def eval_kfold_stub(xvals, yvals):
#     kf = KFold(n_splits = 5, shuffle = True)
#     for train_idxs, test_idxs in kf.split(data):
#         xtrain_this_fold = xvals.loc[train_idxs]
#         xtest_this_fold = xvals.loc[test_idxs]
#         ytrain_this_fold = yvals.loc[train_idxs]
#         ytest_this_fold = yvals.loc[test_idxs]
#         # train a model on this fold
#         # test the model on this fold

