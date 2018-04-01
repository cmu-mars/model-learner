from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from pyDOE import *
import numpy as np
import model


class MLearner:

    def __init__(self, budget, ndim, power_model):
        self.budget = budget
        self.degree = ndim
        self.model = power_model

    def discover(self):
        # performance models has interaction degree of two, based on our study
        model = Pipeline([("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)),
                               ("linear", LinearRegression(fit_intercept=True))])

        # take some ran dom samples
        # this should be replaced with pair wise sampling
        X = np.random.randint(2, size=(self.budget, self.degree))
        y = self.model.evaluateModelFast(X)

        # fit the polynomial model regression
        pmodel = model.fit(X, y)

        return pmodel

    def get_pareto_frontier(self, Xs, Ys, maxX=True, maxY=True):
        # Sort the list in either ascending or descending order of X
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        # Start the Pareto frontier with the first value in the sorted list
        p_front = [myList[0]]
        # Loop through the sorted list
        for pair in myList[1:]:
            if maxY:
                if pair[1] >= p_front[-1][1]:
                    p_front.append(pair)
            else:
                if pair[1] <= p_front[-1][1]:
                    p_front.append(pair)
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]
        return p_frontX, p_frontY

    def get_json(self, pareto_power, pareto_speed):

        data = {}
        data['configurations'] = []
        for i in range(len(pareto_power)):
            data['configurations'].append({
                'config_id': i + 1,
                'power_load': pareto_power[i]/3600*1000,
                'power_load_w': pareto_power[i],
                'speed': pareto_speed[i]
            })

        return data

