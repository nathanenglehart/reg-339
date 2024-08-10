import numpy as np
import pandas as pd
import itertools

from matplotlib import pyplot as plt
from scipy.stats import norm

from lib.gd.logit import logit_regression
from lib.gd.probit import probit_regression

from lib.tools.tools import compute_classification_error_rate
from lib.tools.tools import efron_r_squared
from lib.tools.tools import mcfadden_r_squared

import argparse

# Nathan Englehart (adapted from Autumn 2022)
# todo: add data generation which match conditions under which to use l1 and l2 penalties (right now best to keep these zero)

parser = argparse.ArgumentParser("Binary Choice Regression Driver")
parser.add_argument("-n", help="size of sample to generate", type=int)
parser.add_argument("--verbose", help="run with extended output", action = 'store_true')
parser.add_argument("--l1_penalty", help="l1 penalty", type=int, default = 0)
parser.add_argument("--l2_penalty", help="l2 penalty", type=int, default = 0)
parser.add_argument("--seed", help="seed", type=int)
parser.add_argument("--method", help="probit or logit", type=str)
args = parser.parse_args()

verbose = args.verbose
n = args.n
l1_penalty = args.l1_penalty
l2_penalty = args.l2_penalty
seed = args.seed
method = args.method

np.random.seed(seed)

def driver(method,n,verbose,l1_penalty=0,l2_penalty=0):

    """ Runs univariate and multivariate binary choice regression example simulations using classes in lib/gd 
        
        Args:

            method::[[String]]
                Probit or logit model

            n::[[Int]]
                Number of observations to generate (i.e. rows in X and t)

            verbose::[[Boolean]]
                Run in verbose mode

            l1_penalty::[[Int]]
                L1 penalty to use (optional)

            l2_penalty::[[Int]]
                L2 penalty to use (optional)

    """

    # UNIVARIATE BINARY CHOICE REGRESSION

    x_1 = np.random.normal(0,1,n)
    b_0 = 2
    b_1 = 3
    epsilon = np.random.normal(0,1,n)

    model = probit_regression(l1_penalty = l1_penalty, l2_penalty = l2_penalty)
    if method == "logit": model = logit_regression(l1_penalty = l1_penalty, l2_penalty = l2_penalty)

    t_star = b_0 + b_1 * x_1 + epsilon 
    t = np.where(t_star > 0, 1, 0) 
    model = probit_regression() 

    if(method == 'logit'):
        log_odds = b_0 + b_1 * x_1 + epsilon 
        probabilities = 1 / (1 + np.exp(-log_odds))
        t = np.random.binomial(1, probabilities)
        model = logit_regression() 

    X = np.array([np.ones(n),x_1]).T

    model = model.fit(X,t)
    coef = model.coef_

    t_probs = model.predict_proba(X)
    t_hat = model.predict(X)

    if(verbose):
        print('classification error rate:',compute_classification_error_rate(t,t_hat))
        print('prob preds',t_probs)
        print('McFadden R-Squared:',mcfadden_r_squared(model.coef_, X, t, model))
        print('Efron R-Squared:',efron_r_squared(t,t_probs))
        print('log likelihood',model.log_likelihood(X,t,coef))

        x_1_new, t_probs_new = zip(*sorted(zip(x_1,t_probs)))
        plt.scatter(x_1,t, color='tab:olive')
        plt.plot(x_1_new,t_probs_new,color='tab:cyan')
        plt.savefig('figs/univariate_' + method + '.png')
        plt.xlabel('x_1')
        plt.ylabel('t')
        plt.show()
        plt.close()

    # MULTIVARIATE BINARY CHOICE REGRESSION
	
    x_1 = np.random.normal(0,1,n)
    x_2 = np.random.normal(0,1,n)
    b_0 = 2
    b_1 = 3
    b_2 = 4
    epsilon = np.random.normal(0,1,n)

    model = probit_regression(l1_penalty = l1_penalty, l2_penalty = l2_penalty)
    if method == "logit": model = logit_regression(l1_penalty = l1_penalty, l2_penalty = l2_penalty)

    t_star = b_0 + b_1 * x_1 + b_2 * x_2 + epsilon 
    t = np.where(t_star > 0, 1, 0) 
    model = probit_regression() 

    if(method == "logit"):
        log_odds = b_0 + b_1 * x_1 + b_2 * x_2  + epsilon 
        probabilities = 1 / (1 + np.exp(-log_odds))
        t = np.random.binomial(1, probabilities)

    X = np.array([np.ones(n),x_1,x_2]).T

    model = model.fit(X,t)
    coef = model.coef_

    t_probs = model.predict_proba(X)
    t_hat = model.predict(X)

    if(verbose):
        print('classification error rate:',compute_classification_error_rate(t,t_hat))
        print('prob preds',t_probs)
        print('McFadden R-Squared:',mcfadden_r_squared(model.coef_, X, t, model))
        print('Efron R-Squared:',efron_r_squared(t,t_probs))
        print('log likelihood',model.log_likelihood(X,t,coef))

        x_pts = np.linspace(x_1.min(), x_1.max(), 30)
        y_pts = np.linspace(x_2.min(), x_2.max(), 30)
        x_pairs, y_pairs = np.meshgrid(x_pts,y_pts)

        if(method == 'logit'):
            z = 1 / (1 + np.exp(-(model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs)))
        
        if(method == 'probit'):
            z = norm.cdf(model.coef_[0] + model.coef_[1] * x_pairs + model.coef_[2] * y_pairs)

        fig = plt.figure(figsize = (100,100))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x_pairs,y_pairs,z, rstride=1, cstride=1, color='tab:cyan', alpha=0.4, antialiased=False)
        ax.scatter(x_1,x_2,t, c = 'tab:olive')
        ax.set_ylabel('')
        ax.set_title('t', fontsize=20)
        plt.xlabel('x_1', fontsize=18)
        plt.ylabel('x_2', fontsize=16)
        #plt.savefig('figs/multivariate_' + method + '.png')
        plt.show()
        plt.close()

if __name__ == '__main__':
    driver(method = method, n = n, verbose = verbose, l1_penalty = l1_penalty, l2_penalty = l2_penalty)

