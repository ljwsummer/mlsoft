#!/usr/bin/python
# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2013/02/07 17:09:00"

import optparse
import mlsoft.linear_model.linear_regression as lr

def main():
    parser = optparse.OptionParser()
    parser.add_option('-c', '--config', dest = 'config',
            help = 'config file, you can use examples/model.cfg')
    parser.add_option('-m', '--model', dest = 'model',
            help = 'the model file name that will be saved')

    (options, args) = parser.parse_args()

    # create a new object
    linear_reg = lr.LinearReg(options.config)

    # training a linear regression model
    linear_reg.train()

    # save your model
    linear_reg.save_model(options.model)

    # use linear regression model to predict
    # the param is a m * n matrix
    # it will return a m * 1 matrix
    # m is the number of instance, n is the number of features
    ret = linear_reg.predict(linear_reg.data.X)

if __name__ == '__main__':
    main()

