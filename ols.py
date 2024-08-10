import numpy as np
import pandas as pd
import itertools

from matplotlib import pyplot as plt
from scipy.stats import norm

from lib.gd.ols import ols_regression as gd_ols
from lib.closed.ols import ols_regression as closed_ols

from lib.tools.tools import r_squared
from lib.tools.tools import adj_r_squared

import argparse

# Nathan Englehart (Autumn 2021)
# todo: add data generation which match conditions under which to use l1 and l2 penalties (right now best to keep these zero)

parser = argparse.ArgumentParser("OLS Regression Driver")
parser.add_argument("-n", help="size of sample to generate", type=int)
parser.add_argument("--verbose", help="run with extended output", action = 'store_true')
parser.add_argument("--l1_penalty", help="l1 penalty", type=float, default = 0)
parser.add_argument("--l2_penalty", help="l2 penalty", type=float, default = 0)
parser.add_argument("--seed", help="seed", type=int)
parser.add_argument("--gd", help="gradient descent", action = 'store_true')
args = parser.parse_args()

verbose = args.verbose
n = args.n
l1_penalty = args.l1_penalty
l2_penalty = args.l2_penalty
seed = args.seed
gd = args.gd

ols_regression = closed_ols
if gd: ols_regression = gd_ols

np.random.seed(seed)

def driver(n,verbose,l1_penalty=0,l2_penalty=0):

    """ Runs univariate, multivariate, polynomial, and multivariate polynomial OLS regression example simulations using classes in lib/gd or lib/closed
        
        Args:

            n::[[Int]]
                Number of observations to generate (i.e. rows in X and t)

            verbose::[[Boolean]]
                Run in verbose mode

            l1_penalty::[[Float]]
                L1 penalty to use (optional)

            l2_penalty::[[Float]]
                L2 penalty to use (optional)

    """
   
    # UNIVARIATE REGRESSION

    x_1 = np.random.normal(0,1,n)
    b_0 = 2
    b_1 = 3
    epsilon = np.random.normal(0,1,n)

    model = ols_regression(l2_penalty = l2_penalty)
    if gd: model = ols_regression(l1_penalty = l1_penalty, l2_penalty = l2_penalty)

    t = b_0 + b_1 * x_1 + epsilon 

    X = np.array([np.ones(n),x_1]).T

    model = model.fit(X,t)
    coef = model.coef_

    t_hat = model.predict(X)

    if(verbose):
        print("R-Squared:", r_squared(t,t_hat))
        print("Adjusted R-Squared:", adj_r_squared(t,t_hat,X.shape[1]-1))

        x_1_new, t_hat_new = zip(*sorted(zip(x_1,t_hat)))
        plt.scatter(x_1,t, color='tab:olive')
        plt.plot(x_1_new,t_hat_new,color='tab:cyan')
        plt.savefig('figs/univariate_' + 'ols' + '.png')
        plt.xlabel('x_1')
        plt.ylabel('t')
        plt.show()
        plt.close()

    # POLYNOMIAL REGRESSION

    x_1 = np.random.normal(0,1,n)
    b_0 = 2
    b_1 = 3
    b_2 = 4
    b_3 = 5
    epsilon = np.random.normal(0,1,n)

    model = ols_regression(l2_penalty = l2_penalty)
    if gd: model = ols_regression(l1_penalty = l1_penalty, l2_penalty = l2_penalty)

    t = b_0 + b_1 * x_1 + b_2 * x_1 ** 2 + b_3 * x_1 ** 3 + epsilon

    X = np.array([np.ones(n),x_1,np.square(x_1),np.power(x_1, 3)]).T

    model = model.fit(X,t)
    coef = model.coef_

    t_hat = model.predict(X)

    if(verbose):
        print("R-Squared:", r_squared(t,t_hat))
        print("Adjusted R-Squared:", adj_r_squared(t,t_hat,X.shape[1]-1))

        x_1_new, t_hat_new = zip(*sorted(zip(x_1,t_hat)))
        plt.scatter(x_1,t, color='tab:olive')
        plt.plot(x_1_new,t_hat_new,color='tab:cyan')
        plt.savefig('figs/polynomial_' + 'ols' + '.png')
        plt.xlabel('x_1')
        plt.ylabel('t')
        plt.show()
        plt.close()

    # MULTIVARIATE REGRESSION
	
    x_1 = np.random.normal(0,1,n)
    x_2 = np.random.normal(0,1,n)
    b_0 = 2
    b_1 = 3
    b_2 = 4
    epsilon = np.random.normal(0,1,n)

    t = b_0 + b_1 * x_1 + b_2 * x_2 + epsilon 

    model = ols_regression(l2_penalty = l2_penalty)
    if gd: model = ols_regression(l1_penalty = l1_penalty, l2_penalty = l2_penalty)

    X = np.array([np.ones(n),x_1,x_2]).T

    model = model.fit(X,t)
    coef = model.coef_

    t_hat = model.predict(X)

    if(verbose):
        print("R-Squared:", r_squared(t,t_hat))
        print("Adjusted R-Squared:", adj_r_squared(t,t_hat,X.shape[1]-1))

        x_pts = np.linspace(x_1.min(), x_1.max(), 30)
        y_pts = np.linspace(x_2.min(), x_2.max(), 30)
        x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

        z = model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs 

        fig = plt.figure(figsize = (100,100))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='tab:cyan', alpha=0.4, antialiased=False)
        ax.scatter(x_1,x_2,t, c = 'tab:olive')
        ax.set_ylabel('')
        ax.set_title('t', fontsize=20)
        plt.xlabel('x_1', fontsize=18)
        plt.ylabel('x_2', fontsize=16)
        #plt.savefig('figs/multivariate_' + 'ols' + '.png')
        plt.show()
        plt.close()

if __name__ == '__main__':
    driver(n = n, verbose = verbose, l1_penalty = l1_penalty, l2_penalty = l2_penalty)

