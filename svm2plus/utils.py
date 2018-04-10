'''
Utils for SVM2+
'''

# Author: George Ho <georgeho1618@gmail.com>

import numpy as np


def binarize_targets(y):
    '''
    Converts a target vector to a binary target vector, where y = {-1, +1}

    Parameters
    ----------
    y : array-like, shape (n_samples, )
        Must have only two unique values. Otherwise, a ValueError is raised.
    '''

    if len(np.unique(y)) != 2:
        message = '''Target vector must be a binary vector for binary
        classification. Ensure that target vector has only two unique
        values.'''
        raise ValueError(message)

    y_binarized = y.copy()

    negative_class_target, positive_class_target = np.unique(y)
    y_binarized[y_binarized == positive_class_target] = 1
    y_binarized[y_binarized == negative_class_target] = -1

    return y_binarized, positive_class_target, negative_class_target


def unbinarize_targets(y_binarized,
                       positive_class_target,
                       negative_class_target):
    '''
    Converts a binary target vector back to the original target vector

    Parameters
    ----------
    y_binarized : array-like, shape (n_samples, )
        Must have only two unique values: -1 and +1. Otherwise, a ValueError is
        raised.

    positive_class_target : int
        Target value to denote membership in the positive class.
        
    negative_class_target : int
        Target value to denote membership in the negative class.
    '''

    assert set(np.unique(y_binarized)).issubset(set([-1, 1])), \
        'Binarized target vector must have only two unique values: -1 and +1'

    y = y_binarized.copy()

    y[y == 1] = positive_class_target
    y[y == -1] = negative_class_target

    return y
