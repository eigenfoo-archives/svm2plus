'''
Benchmark SVC2Plus against sklearn's SVC using the synthetic dataset described
by Vapnik (Intelligent Teacher).

Simply run this script to produce two txt files:
    1. svc_results.txt
    2. svc2plus_results.txt

which contain the error rates of the SVC and SVC2Plus on the synthetic dataset.
'''

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), '../svm2plus/'))

from sklearn.svm import SVC
from svm2plus import SVC2Plus
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import zero_one_loss


TRAIN_DIR = '../data/synthetic_data/train/'
TEST_DIR = '../data/synthetic_data/test/'

# Grid search as described by Vapnik
PARAM_GRID = {'C': np.exp2(np.linspace(-5, 5, 21)),
              'gamma': np.exp2(np.linspace(-6, 6, 25))}

'''
with open('svc_results.txt', 'w') as f:
    for l in [20, 40, 80]:                # Loop over all training sizes
        for epsilon in [0.01, 0.1, 1.0]:  # Loop over all noise parameters
            print('[l={}, epsilon={}]'.format(l, epsilon), file=f)
            errs = np.zeros(10)

            for n in range(10):           # Loop over all trials
                # Read in training and test sets
                train = pd.read_csv(
                    TRAIN_DIR +
                    'train_{}_{}_{}.csv'.format(l,
                                                epsilon,
                                                n),
                    index_col=0)
                test = pd.read_csv(
                    TEST_DIR +
                    'test_{}_{}_{}.csv'.format(l,
                                               epsilon,
                                               n),
                    index_col=0)

                # 6-fold cross validation, as described by Vapnik
                cv = GridSearchCV(SVC(), PARAM_GRID, scoring='accuracy', cv=6)
                cv.fit(train.loc[:, 'x1':'x2'], train.loc[:, 'label'])
                clf = cv.best_estimator_
                y_pred = clf.predict(test.loc[:, 'x1':'x2'])

                # Compute error rate
                err = zero_one_loss(test.loc[:, 'label'], y_pred)

                # Write error rate
                msg = '[n={}] Error rate = {}'.format(n, err)
                print(msg, file=f)
                errs[n] = err

            # Write average error rate
            print('Average error rate = {}\n'.format(np.mean(errs)), file=f)
'''

with open('svc2plus_results.txt', 'w') as f:
    for l in [20, 40, 80]:                # Loop over all training sizes
        for epsilon in [0.01, 0.1, 1.0]:  # Loop over all noise parameters
            print('[l={}, epsilon={}]'.format(l, epsilon), file=f)
            errs = np.zeros(10)

            for n in range(10):           # Loop over all trials
                # Read in training and test sets
                train = pd.read_csv(
                    TRAIN_DIR +
                    'train_{}_{}_{}.csv'.format(l,
                                                epsilon,
                                                n),
                    index_col=0)
                test = pd.read_csv(
                    TEST_DIR +
                    'test_{}_{}_{}.csv'.format(l,
                                               epsilon,
                                               n),
                    index_col=0)

                # 6-fold cross validation, as described by Vapnik
                cv = GridSearchCV(SVC2Plus(), PARAM_GRID, scoring='accuracy', cv=6)
                cv.fit(train.loc[:, 'x1':'x2'], train.loc[:, 'label'])
                clf = cv.best_estimator_
                y_pred = clf.predict(test.loc[:, 'x1':'x2'])

                # Compute error rate
                err = zero_one_loss(test.loc[:, 'label'], y_pred)

                # Write error rate
                msg = '[n={}] Error rate = {}'.format(n, err)
                print(msg, file=f)
                errs[n] = err

            # Write average error rate
            print('Average error rate = {}\n'.format(np.mean(errs)), file=f)
