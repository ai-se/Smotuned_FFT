from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import pandas as pd

def cut_position(pos, neg, percentage=0):
    return int(pos["bug"].count() * percentage / 100), int(neg["bug"].count() * percentage / 100)

def divide_train_test(pos, neg, cut_pos, cut_neg):
    data_train = pd.concat([pos.iloc[:cut_pos,:], neg.iloc[:cut_neg,:]],ignore_index=True)
    data_test = pd.concat([pos.iloc[cut_pos:,:], neg.iloc[cut_neg:,:]], ignore_index=True)
    return data_train, data_test

def split_two(corpus):
    pos = corpus[corpus['bug']==1]
    neg = corpus[corpus['bug'] != 1]
    return {'pos': pos, 'neg': neg}