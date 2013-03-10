#!/usr/bin/python
# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2013/03/10 15:29:00"

import optparse
import mlsoft.linear_model.logistic_regression as logr

def main():
    parser = optparse.OptionParser()
    parser.add_option('-c', '--config', dest = 'config',
            help = 'config file, you can use examples/model.cfg')
    parser.add_option('-m', '--model', dest = 'model',
            help = 'the model file name that will be saved')

    (options, args) = parser.parse_args()

    # create a new object
    logistic_reg = logr.LogisticReg(options.config)

    # training a logistic regression model
    logistic_reg.train()

    # save your model
    logistic_reg.save_model(options.model)

    # use linear regression model to predict
    # the param is a m * n matrix
    # it will return a m * 1 matrix
    # m is the number of instance, n is the number of features
    ret = logistic_reg.predict(logistic_reg.data.X)

if __name__ == '__main__':
    main()

