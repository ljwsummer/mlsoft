# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2013/02/22 12:03"

import numpy as np
from linear_model import LinearModel

class LogisticReg(LinearModel):
    """logistic regression"""

    def __init__(self, conf_file):
        LinearModel.__init__(self, conf_file)
        self.alpha = self.conf.getfloat('logistic_regression', 'alpha')
        self.iters = self.conf.getint('logistic_regression', 'num_iters')
        self.lambd = self.conf.getfloat('logistic_regression', 'lambda')

    def _compute_cost(self, X, Y, theta):
        pass

    def _compute_gradient(self, X, Y, theta):
        pass

    def _gradient_descent(self):
        grad = self._compute_gradient(self.data.X, self.data.Y, self.theta)
        self.theta -= self.alpha * grad / self.m

    def predict(self):
        pass

    def save_model(self, model_file):
        pass

    def load_model(self, model_file):
        pass



