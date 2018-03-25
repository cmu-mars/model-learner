import numpy as np
import mlearner
import model
import os
import itertools
import json

from learner.mlearner import MLearner
from learner.model import genModelTermsfromString, Model

path = '../conf/'
model = 'model.txt'
config_list_file = 'config_list1.json'
ndim = 20
nrow = 1
budget = 2000
test_size = 10000
mu, sigma = 0, 0.1
default_conf = np.reshape(np.ones(ndim), (1, ndim))

# learner = mlearner.MLearner(10, 3, ("x", "y"), [[0, 0], [5, 10]], "func.xml")
# learned_model = learner.discover()
# coefficients = learned_model.named_steps['linear'].coef_
# print(coefficients)

with open(os.path.join(path, model), 'r') as model_file:
    model_txt = model_file.read()

power_model_terms = genModelTermsfromString(model_txt)
power_model = Model(power_model_terms, ndim)

# xTrain = np.ones(ndim)
# xTrain = np.reshape(xTrain, (nrow, ndim))
# yTrain = power_model.evaluateModelFast(xTrain)
# print(yTrain)

# learn the model
learner = MLearner(budget, ndim, power_model)
learned_model = learner.discover()

print(learned_model.named_steps['linear'].coef_)

# configs = itertools.product(range(2), repeat=ndim)
# xTest = np.zeros(shape=(2**ndim, ndim))
# i = 0
# for c in configs:
#     xTest[i, :] = np.array(c)
#     i += 1

xTest = np.random.randint(2, size=(test_size, ndim))
yTestPower = learned_model.predict(xTest)

# adding noise for the speed
s = np.random.normal(mu, sigma, test_size)
yTestSpeed = yTestPower/100 + s

yDefaultPower = learned_model.predict(default_conf)
yDefaultSpeed = yDefaultPower/100

conf_pareto_front = learner.get_pareto_frontier(yTestPower, yTestSpeed, maxX=False, maxY=True)
json_data = learner.get_json(conf_pareto_front[0], conf_pareto_front[1])


# add the default configuration
json_data['configurations'].append({
    'config_id': 0,
    'power_load': yDefaultPower[0]/3600*1000,
    'power_load_wh': yDefaultPower[0],
    'speed': yDefaultSpeed[0]
})

with open(os.path.join(path, config_list_file), 'w') as outfile:
    json.dump(json_data, outfile)
