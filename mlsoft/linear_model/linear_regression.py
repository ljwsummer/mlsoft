# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2012/12"

import numpy as np
from linear_model import LinearModel

class LinearReg(LinearModel):
    """linear regression"""

    def __init__(self, conf_file):
        LinearModel.__init__(self, conf_file)
        if 'train_data' in self.conf.options('linear_regression'):
            self._load_data(self.conf.get('linear_regression', 'train_data'))
        self.alpha = self.conf.getfloat('linear_regression', 'alpha')
        self.iters = self.conf.getint('linear_regression', 'num_iters')
        self.lambd = self.conf.getfloat('linear_regression', 'lambda')
        self.intercept = self.conf.getboolean('linear_regression', 'has_intercept')

    def _compute_cost(self, X, Y, theta):
        m = X.shape[0]
        res = X * theta - Y
        idx = 1 if self.intercept else 0
        j = sum(np.power(res, 2)) / m + self.lambd * sum(np.power(theta[idx:], 2)) / (2 * m)
        return j[0,0]

    def _compute_gradient(self, X, Y, theta):
        if self.intercept:
            ret = X.T * (X * theta - Y) + np.vstack([np.matrix('0'), self.lambd * theta[1:]])
        else:
            ret = X.T * (X * theta - Y) + self.lambd * theta
        return ret


