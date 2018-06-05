from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import statsmodels.api as sm


def sample_random(ndim, budget):
    # take some ran dom samples
    # this should be replaced with pair wise sampling
    X = np.random.randint(2, size=(budget, ndim))
    return X


def learn_with_interactions(X, y):
    # performance models has interaction degree of two, based on our study
    model = Pipeline([("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)),
                           ("linear", LinearRegression(fit_intercept=True))])

    # fit the polynomial model regression
    pmodel = model.fit(X, y)

    return pmodel


def learn_without_interactions(X, y):
    # performance models has interaction degree of two, based on our study
    model = Pipeline([("poly", PolynomialFeatures(degree=1, include_bias=True)),
                      ("linear", LinearRegression(fit_intercept=True))])

    # fit the polynomial model regression
    pmodel = model.fit(X, y)

    return pmodel


def get_pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    # Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    idx_sorted = sorted(range(len(Xs)), key=lambda k: Xs[k])
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
    i = 0
    pareto_idx = [idx_sorted[i]]
    # Loop through the sorted list
    for pair in myList[1:]:
        i += 1
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
                pareto_idx.append(idx_sorted[i])
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
                pareto_idx.append(idx_sorted[i])
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return pareto_idx, p_frontX, p_frontY


def mean_absolute_percentage_error(y_true, y_pred):
    # eps for avoiding division by zero
    return np.mean(np.abs((y_true - y_pred) / y_true + np.finfo(float).eps)) * 100


def stepwise_feature_selection(X, y,
                               initial_list=[],
                               threshold_in=0.01,
                               threshold_out=0.05,
                               verbose=True):
    ndim = X.shape[1]
    features = [i for i in range(ndim)]
    included = list(initial_list)

    while True:
        changed = False
        # forward
        excluded = list(set(features) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_feature in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[:, included + [new_feature]]))).fit()
            new_pval[new_feature] = model.pvalues[1]
        best_pval = new_pval.min()

        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add {:30} with p-value {:.6}'.format("o" + str(best_feature), best_pval))

        # backward
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[:, included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()

        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(included[worst_feature])
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format("o" + str(worst_feature), worst_pval))

        if not changed:
            break

    return included
