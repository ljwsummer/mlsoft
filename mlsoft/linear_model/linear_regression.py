# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2012/12"

import ConfigParser
import numpy as np
import mlsoft.common.reader

class LinearReg(object):
    """linear regression"""

    def __init__(self, conf_file):
        self._load_conf(conf_file)
        self.alpha = self.conf.getfloat('linear_regression', 'alpha')
        self.iters = self.conf.getint('linear_regression', 'num_iters')
        self.lambd = self.conf.getfloat('linear_regression', 'lambda')
        data_file = self.conf.get('global', 'train_data')
        self._load_data(data_file)
        self.m = self.data.X.shape[0]
        self.n = self.data.X.shape[1]
        self.theta = np.mat(np.zeros((self.n, 1)))
        self.costs = []

    def _load_conf(self, conf_file):
        self.conf = ConfigParser.ConfigParser()
        self.conf.optionxform = str
        self.conf.read(conf_file)

    def _load_data(self, data_file):
        self.data = mlsoft.common.reader.Data(data_file)

    def _compute_cost(self, X, Y, theta):
        m = X.shape[0]
        res = X * theta - Y
        j = sum(np.power(res, 2)) / m
        return j[0,0]

    def _compute_gradient(self, X, Y, theta):
        return X.T * (X * theta - Y) + self.lambd * theta

    def _gradient_descent(self):
        grad = self._compute_gradient(self.data.X, self.data.Y, self.theta)
        self.theta -= self.alpha * grad / self.m

    def train(self):
        for x in xrange(self.iters):
            self._gradient_descent()
            cost = self._compute_cost(self.data.X, self.data.Y, self.theta)
            self.costs.append(cost)

    def predict(self, X):
        delta = X.shape[1] - self.theta.shape[0]
        if delta < 0:
            theta = self.theta[:X.shape[1]]
        elif delta > 0:
            theta = np.vstack([self.theta, np.mat(np.zeros((delta, 1)))])
        else:
            theta = self.theta
        return X * theta

    def save_model(self, model_file):
        fp = open(model_file, 'w')
        t = [str(x) for x in self.theta.flat]
        print >> fp, ';'.join(t)
        fp.close()

    def load_model(self, model_file):
        fp = open(model_file)
        self.theta = np.matrix(fp.readline().strip())
        fp.close()


