from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import os
from collections import OrderedDict
from Learners import *
from demos import cmd
from DE import DE
import time

## Global Bounds
cwd = os.getcwd()
data_path = os.path.join(cwd, "../Data")
data = {"ivy": ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"], \
        "lucene": ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"], \
        "poi": ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"], \
        "synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"], \
        "velocity": ["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"], \
        "camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"], \
        "jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"], \
        "log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"], \
        "xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"], \
        "xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
        }
learners_para_dic=[OrderedDict([("m",1), ("r",1),("k",1), ("max_depth",1)])]
learners_para_bounds=[[(50,100,200, 400), (1,6), (5,21), (1,20), (1,50)]]
learners_para_categories=[["categorical", "integer", "integer"]]
learners=[DT, RF, SVM, NB, KNN, LR]
measures=["d2h", "popt20"]
repeats=30

def _test(res=''):
    paths = [os.path.join(data_path, file_name) for file_name in data[res]]
    train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
    test_df = pd.read_csv(paths[-1])

    ### getting rid of first 3 columns
    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
    test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)

    final_dic={}
    for i in learners:
        temp={}
        for x in measures:
            l=[]
            start_time = time.time()
            for _ in xrange(repeats):
                ## Shuffle
                train_df = train_df.sample(frac=1).reset_index(drop=True)
                test_df = test_df.sample(frac=1).reset_index(drop=True)

                if x=="d2h":
                    de = DE(GEN=5, Goal="Min")
                    v, pareto = de.solve(main, OrderedDict(learners_para_dic[0]),
                                         learners_para_bounds[0], learners_para_categories[0], i, x, train_df)
                    paras=v.ind
                    data_train = smote.execute(paras.values(), samples=train_df[:, :-1], labels=train_df[:, -1:])

                    labels=i(data_train[:, :-1].values, data_train[:, -1:].values, test_df[:, :-1].values)
                    val=evaluation(x, labels, test_df[:, -1:].values, test_df)
                    l.append(val)
                elif x== "popt20":
                    de = DE(GEN=5, Goal="Max")
                    v, pareto = de.solve(i, OrderedDict(learners_para_dic[0]),
                                         learners_para_bounds[0], learners_para_categories[0], i, x, train_df)
                    paras = v.ind
                    data_train = smote.execute(paras.values(), samples=train_df[:, :-1], labels=train_df[:, -1:])

                    labels = i(data_train[:, :-1].values, data_train[:, -1:].values, test_df[:, :-1].values)
                    val = evaluation(x, labels, test_df[:, -1:].values, test_df)
                    l.append(val)
            total_time=time.time() - start_time
            temp[x]=[l,total_time]
        final_dic[i.__name__]=temp

if __name__ == '__main__':
    eval(cmd())