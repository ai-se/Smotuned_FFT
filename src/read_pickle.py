from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import os
import pickle
import numpy as np

path=os.getcwd()
path=os.path.join(path+"/../dump/late/")

print(path)
temp={}
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        a = os.path.join(root, name)
        with open(a, 'rb') as handle:
            temp[name.split("_")[0]] = pickle.load(handle)

## Values
for filename,values in temp.iteritems():
    for lea, val in values.iteritems():
        for mea, v in val.iteritems():
            print(filename,lea, mea,end=" ")
            print(np.median(v[0]))

## runtimes
l=[]
for filename,values in temp.iteritems():
    for lea, val in values.iteritems():
        for mea, v in val.iteritems():
            print(filename,lea, mea,end=" ")
            l.append(round(v[2]/30,2))
            print(round(v[2]/30,2))
print(np.median(l))
