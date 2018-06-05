from learner.mlearner import learn_with_interactions, learn_without_interactions, mean_absolute_percentage_error, sample_random, stepwise_feature_selection
from learner.model import genModelTermsfromString, Model, genModelfromCoeff

ndim = 20
true_model = """10 + 1.00 * o0 + 2.00 * o1 + 3.00 * o2 +
4.00 * o3 + 5.00 * o4 + 6.00 * o5 + 7.00 * o6 + 8.00 * o7 + 
1.00 * o8 + 2.00 * o9 + 3.00 * o10 + 4.00 * o11 + 5.00 * o12 + 
6.00 * o13 + 7.00 * o14 + 8.00 * o15 + 1.00 * o16 + 2.00 * o17 + 
3.00 * o18 + 4.00 * o19 + 1 * o0 * o1 + 3 * o3 * o6"""
model_terms = genModelTermsfromString(true_model)
true_model = Model(model_terms, ndim)
X = sample_random(ndim=ndim, budget=200)
y = true_model.evaluateModelFast(X)
stepwise_feature_selection(X, y)
