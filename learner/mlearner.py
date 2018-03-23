from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from pyDOE import *
import numpy as np
from polyparser.func_util import function_utilities


class MLearner:

    def __init__(self, budget, degree, vars, domain, model_filename):
        self.budget = budget
        self.degree = degree
        self.vars = vars
        self.domain = domain
        self.funcfile = model_filename

    def discover(self):
        model = Pipeline([("poly", PolynomialFeatures(degree=self.degree)),
                          ("linear", LinearRegression(fit_intercept=False))])

        L = len(self.vars)
        # Latin Hypercube Design, we may change this later
        design = lhs(L, samples=self.budget, criterion="center")
        # scale into feature range
        LD = len(design)
        X = np.zeros((LD, L))
        for i in range(L):
            X[:, i] = design[:, i] * (self.domain[1][i] - self.domain[0][i]) + self.domain[0][i]

        eval = function_utilities.Evaluator(self.funcfile)

        y = np.zeros((LD,1))
        for i in range(LD):
            y[i] = eval.eval_function(X[i, :])

        # fit the polynomial model regression
        model = model.fit(X, y)

        return model

