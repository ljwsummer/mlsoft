# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2013/02/22 12:03"

import math
import numpy as np
from linear_model import LinearModel

class LogisticReg(LinearModel):
    """logistic regression"""

    def __init__(self, conf_file):
        LinearModel.__init__(self, conf_file)
        self.alpha = self.conf.getfloat('logistic_regression', 'alpha')
        self.iters = self.conf.getint('logistic_regression', 'num_iters')
        self.lambd = self.conf.getfloat('logistic_regression', 'lambda')
        self.intercept = self.conf.getboolean('linear_regression', 'has_intercept')

    def _sigmoid(self, z):
        return 1 / (1 + np.power(math.e, -z))

    def _compute_cost(self, X, Y, theta):
        m = X.shape[0]
        res = - (np.multiply(Y, np.log(self._sigmoid(X * theta)))
                + np.multiply(1 - Y, np.log(1 - Y, self._sigmoid(X * theta))))
        idx = 1 if self.intercept else 0
        reg = self.lambd * sum(np.power(theta[idx:], 2)) / (2 * m)
        j = sum(res) / m + reg
        return j[0,0]

    def _compute_gradient(self, X, Y, theta):
        if self.intercept:
            ret = X.T * (self._sigmoid(X * theta) - Y) + np.vstack([np.matrix('0'), self.lambd * theta[1:]])
        else:
            ret = X.T * (self._sigmoid(X * theta) - Y) + self.lambd * theta
        return ret

    def predict(self, X):
        ret = LinearModel.predict(self. X)
        return self._sigmoid(ret)


