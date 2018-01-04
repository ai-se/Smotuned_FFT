from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
from sklearn.metrics import auc


def subtotal(x):
  xx = [0]
  for i, t in enumerate(x):
    xx += [xx[-1] + t]
  return xx[1:]

def get_recall(true):
  total_true = float(len([i for i in true if i == 1]))
  hit = 0.0
  recall = []
  for i in xrange(len(true)):
    if true[i] == 1:
      hit += 1
    recall += [hit / total_true if total_true else 0.0]
  return recall

# pass the whole testing data with label
def get_popt(data):
  data.sort_values(by=["bug", "loc"], ascending=[0, 1], inplace=True)
  x_sum = float(sum(data['loc']))
  x = data['loc'].apply(lambda t: t / x_sum)
  xx = subtotal(x)

  # get  AUC_optimal
  yy = get_recall(data['bug'].values)
  xxx = [i for i in xx if i <= 0.2]
  yyy = yy[:len(xxx)]
  s_opt = round(auc(xxx, yyy), 3)

  # get AUC_worst
  xx = subtotal(x[::-1])
  yy = get_recall(data['bug'][::-1].values)
  xxx = [i for i in xx if i <= 0.2]
  yyy = yy[:len(xxx)]
  try:
    s_wst = round(auc(xxx, yyy), 3)
  except:
    # print "s_wst forced = 0"
    s_wst = 0

  # get AUC_prediction
  data.sort_values(by=["prediction", "loc"], ascending=[0, 1], inplace=True)
  x = data['loc'].apply(lambda t: t / x_sum)
  xx = subtotal(x)
  yy = get_recall(data['bug'].values)
  xxx = [k for k in xx if k <= 0.2]
  yyy = yy[:len(xxx)]
  try:
    s_m = round(auc(xxx, yyy), 3)
  except:
    return 0

  Popt = (s_m - s_wst) / (s_opt - s_wst)
  return round(Popt,2)