import copy
import json
import operator

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

def entropy(y):
    val_count = y.value_counts()
    probs = map(lambda count: count / len(y), val_count)
    e = sum(map(entropy_helper, probs))
    return e

def entropy_helper(prob):
    return - prob * np.log2(prob)

def gain(y,x):
    weighted_entropies = map(lambda xVal: gain_helper(y, x, xVal), x.unique())
    g = sum(weighted_entropies)
    return entropy(y) - g

def gain_helper(y, x, xVal):
    rows = x == xVal
    e = entropy(y[rows])
    weighted_e = e * np.count_nonzero(rows) / len(x)
    return weighted_e

def gain_ratio(y,x):
    g = gain(y,x)
    return g / entropy(y) if entropy(y) != 0 else 0


def select_split(X,y):
    grs = X.apply(lambda col: gain_ratio(y,col), axis=0)
    grs_tuples = grs.items()
    return max(grs_tuples,key=operator.itemgetter(1))

def make_tree(X,y):
    tree = {}
    if entropy(y) == 0: # All examples have same label
        tree = y.unique()[0]
    elif len(X.columns) == 0 or no_unique_cols(X): # No labels left to test
        tree = y.mode()[0]
    else:
        col, gr = select_split(X,y)
        tree[col] = {colVal: make_tree_helper(X, y, col, colVal) for colVal in X[col].unique()}
    return tree

def make_tree_helper(X, y, col, colVal):
    rows = X[col] == colVal
    cut_y = y[rows]
    cut_X = X.copy().drop(col, axis=1)[rows]
    return make_tree(cut_X, cut_y)

def no_unique_cols(X):
    return X.apply(lambda col: col.nunique() == 1, axis=0).all()

# if you want to print like me :)
def print_tree(tree):
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            if type(tree) == np.int64:
                return int(tree)
            if type(tree) == np.float64:
                return float(tree)
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            elif type(key) == np.bool8:
                new_tree[bool(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))

def generate_rules(tree):
    if not isinstance(tree, dict):
        return [[tree]]
    
    rules = []
    colName = next(iter(tree))
    for colVal in tree[colName]:
        future_rules = generate_rules(tree[colName][colVal])
        for i in range(len(future_rules)):
            future_rules[i].insert(0, (colName, colVal))
        rules.extend(future_rules)
    return rules

def select_split2(X,y):
    no_numeric_X = X.copy()
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            if X[col].nunique() > 1:
                no_numeric_X = pd.concat([no_numeric_X, numeric_to_categorical(X[col])], axis=1)
            no_numeric_X.drop(col, axis=1, inplace=True)
    col, gr = select_split(no_numeric_X, y)
    return col, gr

def numeric_to_categorical(col):
    vals = np.sort(col.unique())
    splits = [np.mean((vals[i], vals[i + 1])) for i in range(len(vals) - 1)]
    cols = map(lambda split: (col < split).rename(f"{col.name}<{split:.2f}"), splits)
    return pd.concat(cols, axis=1)

def make_tree2(X,y,min_split_count=5):
    tree = {}
    if entropy(y) == 0: # All examples have same label
        tree = y.unique()[0]
    elif len(X.columns) == 0 or no_unique_cols(X) or len(X) < min_split_count: # No labels left to test
        tree = y.mode()[0]
    else:
        col, gr = select_split2(X,y)
        if '<' in col:
            splitCol, splitVal = col.split("<")
            X.insert(len(X.columns), col, X[splitCol] < float(splitVal))
            X.drop(splitCol, axis=1, inplace=True)
        tree[col] = {str(colVal): make_tree_helper2(X, y, col, colVal, min_split_count) for colVal in X[col].unique()}
    return tree

def make_tree_helper2(X, y, col, colVal,min_split_count):
    rows = X[col] == colVal
    cut_y = y[rows]
    cut_X = X.copy().drop(col, axis=1)[rows]
    return make_tree2(cut_X, cut_y, min_split_count)

def make_prediction(rules,x,default):
    relevantRules = rules
    i = 0
    while len(relevantRules):
        nextRule = relevantRules[0][i]
        if type(nextRule) is tuple:
            nextCol = nextRule[0]
            relevantRules = list(filter(lambda ruleSet: followsRule(ruleSet[i], nextCol, x), relevantRules))
        else:
            prediction = nextRule
            return prediction
        i += 1
    return(default)

def followsRule(rule, col, x):
    xVal = None
    if "<" in col:
        splitCol, splitVal = col.split("<")
        xVal = str(x[splitCol] < float(splitVal))
    else:
        xVal = x[col]
    return rule == (col, xVal)

def binned(y, binSize):
    return (y//binSize*binSize).astype(int)
