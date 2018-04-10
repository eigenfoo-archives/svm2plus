'''
SVM2+
'''

# Author: George Ho <georgeho1618@gmail.com>

import numpy as np
import utils
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.metrics.pairwise import (rbf_kernel,
                                      linear_kernel,
                                      polynomial_kernel,
                                      sigmoid_kernel)


class SVC2Plus(BaseEstimator):
    '''
    SVM2+ for binary classification.

    This implementation is based on scikit-learn's `SVC` class.

    Parameters
    ----------
    lmbda : float
        Regularization parameter.

    decision_kernel : 'rbf', 'linear', 'poly' or 'sigmoid'
        Kernel in the decision space.

    correcting_kernel : 'rbf', 'linear', 'poly' or 'sigmoid'
        Kernel in the correcting space.

    All other parameters are identical to those of sklearn.svm.SVC.

    Attributes
    ----------
    _positive_class_target : int
        Target value to denote membership in the positive class.

    _negative_class_target : int
        Target value to denote membership in the negative class.

    All other attributes are identical to those of sklearn.svm.SVC.

    References
    ----------
    Xu, Xinxing & Tianyi Zhou, Joey & Tsang, Ivor & Qin, Zheng & Siow Mong Goh,
    Rick & Liu, Yong. (2016). Simple and Efficient Learning using Privileged
    Information.
    '''

    def __init__(self, lmbda=1.0, C=1.0, decision_kernel='rbf',
                 correcting_kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False, tol=0.001,
                 cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', random_state=None):
        self.lmbda = lmbda
        self.C = C
        self.decision_kernel = decision_kernel
        self.correcting_kernel = correcting_kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

        self.support_ = None
        self.support_vectors_ = None
        self.n_support_ = None
        self.dual_coef_ = None
        self.intercept_ = None

        # Because SVM2+ relies on the positive and negative class target values
        # being being 1 and -1, respectively, it is good to store these.
        self._positive_class_target = None
        self._negative_class_target = None

    def svm2plus_kernel(self, X, y, Z):
        '''
        Computes kernel (a.k.a. Gram matrix) for SVM2+, assuming a squared
        hinge loss. For more information, see Xu et al. (2016).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix to compute SVM2+ kernel matrix for. If we wish to use
            a basis function (e.g. rbf, poly), we must transform X _prior_ to
            passing it to this function!

        Z : array-like, shape (n_samples, n_privileged_features)
            Privileged information.

        y : array-like, shape (n_samples, )
            Correct targets.

        lmbda, C : floats
            Regularization parameters.
        '''

        n = X.shape[0]
        X = X.reshape(X.shape[0], -1)
        Z = Z.reshape(Z.shape[0], -1)
        y, _, _ = utils.binarize_targets(y)

        if self.decision_kernel == 'rbf':
            K = rbf_kernel(X, X, gamma=self.gamma)
        elif self.decision_kernel == 'linear':
            K = linear_kernel(X, X)
        elif self.decision_kernel == 'poly':
            K = polynomial_kernel(X, X,
                                  degree=self.degree,
                                  gamma=self.gamma,
                                  coef0=self.coef0)
        elif self.decision_kernel == 'sigmoid':
            K = sigmoid_kernel(X, X,
                               gamma=self.gamma,
                               coef0=self.coef0)
        else:
            raise ValueError('''Kernel in decision space must be one of 'rbf',
                             'linear', 'poly' or 'sigmoid'.''')

        if self.correcting_kernel == 'rbf':
            K_tilde = rbf_kernel(Z, Z, gamma=self.gamma)
        elif self.correcting_kernel == 'linear':
            K_tilde = linear_kernel(Z, Z)
        elif self.correcting_kernel == 'poly':
            K_tilde = polynomial_kernel(Z, Z,
                                        degree=self.degree,
                                        gamma=self.gamma,
                                        coef0=self.coef0)
        elif self.correcting_kernel == 'sigmoid':
            K_tilde = sigmoid_kernel(Z, Z,
                                     gamma=self.gamma,
                                     coef0=self.coef0)
        else:
            raise ValueError('''Kernel in correcting space must be one of 'rbf',
                             'linear', 'poly' or 'sigmoid'.''')

        Q_lmbda = 1/self.lmbda * (K_tilde
                                  - np.dot(
                                      np.dot(K_tilde,
                                             np.linalg.inv(
                                                 (self.lmbda/self.C)
                                                 * np.identity(n)
                                                 + K_tilde)),
                                      K_tilde))

        return K + np.multiply(Q_lmbda,
                               np.outer(y, y.T))

    def fit(self, X, y, Z):
        '''
        Fits SVM2+ model to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples, )
            Targets of training data.

        Z : array-like, shape (n_samples, n_privileged_features)
            Privileged information.
        '''

        X = X.reshape(X.shape[0], -1)
        Z = Z.reshape(Z.shape[0], -1) 
        y_binarized, self._positive_class_target, \
            self._negative_class_target = utils.binarize_targets(y)

        if self.gamma == 'auto':
            self.gamma = 1.0 / X.shape[1]
        else:
            self.gamma = self.gamma

        # The parameters `degree`, `gamma` and `coef0` are irrelevant here, as
        # the kernel is precomputed, and is not linear, poly, sigmoid or rbf.
        clf = SVC(C=self.C, kernel='precomputed', shrinking=self.shrinking,
                  probability=self.probability, tol=self.tol,
                  cache_size=self.cache_size, class_weight=self.class_weight,
                  verbose=self.verbose, max_iter=self.max_iter,
                  decision_function_shape=self.decision_function_shape,
                  random_state=self.random_state)

        # Pass in precomputed Gram matrix, not data
        clf.fit(self.svm2plus_kernel(X, y_binarized, Z), y)

        self.support_ = clf.support_
        self.support_vectors_ = X[clf.support_]
        self.n_support_ = clf.n_support_
        self.dual_coef_ = clf.dual_coef_
        self.intercept_ = clf.intercept_

    def predict(self, X):
        '''
        Classifies new samples using the SVM2+ coefficients.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New samples to classify.

        dual_coef : array-like, shape (n_support_vectors, )
            Dual coefficients of support vectors in trained model.
            In the literature, this is alpha * y.

        support_vectors : array-like, shape (n_support_vectors, n_features)
            Support vectors.
        '''

        X = X.reshape(X.shape[0], -1)

        if self.decision_kernel == 'rbf':
            kernel = rbf_kernel(X, self.support_vectors_, gamma=self.gamma)
        elif self.decision_kernel == 'linear':
            kernel = linear_kernel(X, self.support_vectors_)
        elif self.decision_kernel == 'poly':
            kernel = polynomial_kernel(X, self.support_vectors_,
                                       degree=self.degree,
                                       gamma=self.gamma,
                                       coef0=self.coef0)
        elif self.decision_kernel == 'sigmoid':
            kernel = sigmoid_kernel(X, self.support_vectors_,
                                    gamma=self.gamma,
                                    coef0=self.coef0)
        else:
            raise ValueError('''Kernel in decision space must be one of 'rbf',
                             'linear', 'poly' or 'sigmoid'.''')

        predictions = np.dot(self.dual_coef_, kernel.T) + self.intercept_

        predictions = np.sign(predictions).flatten()
        predictions = utils.unbinarize_targets(predictions,
                                               self._positive_class_target,
                                               self._negative_class_target)

        return predictions
