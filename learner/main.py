import numpy as np
import mlearner
import model
import os

from learner.model import genModelTermsfromString, Model

path = '../conf/'
model = 'model.txt'
ndim = 20

# learner = mlearner.MLearner(10, 3, ("x", "y"), [[0, 0], [5, 10]], "func.xml")
# learned_model = learner.discover()
# coefficients = learned_model.named_steps['linear'].coef_
# print(coefficients)

with open(os.path.join(path, model), 'r') as model_file:
    model_txt = model_file.read()

power_model_terms = genModelTermsfromString(model_txt)
power_model = Model(power_model_terms, ndim)
xTest = np.ones(ndim)

yTest = power_model.evaluateModelFast(xTest)

print(yTest)
