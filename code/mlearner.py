from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from pyDOE import *
import numpy as np
from polyparser.func_util import function_utilities


class mlearner():

    def __init__(self,budget,degree,vars,domain,model_filename):
        self.budget = budget
        self.degree = degree
        self.vars = vars
        self.domain = domain
        self.funcfile = model_filename

    def learner(self):
        model = Pipeline([("poly", PolynomialFeatures(degree=self.degree)),
                          ("linear", LinearRegression(fit_intercept=False))])

        L = len(self.vars)
        # Latin Hypercube Design, we may change this later
        design = lhs(L, samples=self.budget, criterion="center")
        # scale into feature range
        for i in range(L):
            X[:,i] = design[:,i]*(domain[1,i]-domain[0,i]) + domain[0,i]

        eval = function_utilities.Evaluator(self.funcfile)
        y = eval.eval_function(X)

        # fit the polynomial model regression
        model = model.fit(X, y)

        return model





