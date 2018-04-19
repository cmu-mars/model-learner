import numpy as np
import os
import itertools
import json

# the main for learning

from mlearner import MLearner
from model import genModelTermsfromString, Model, genModelfromCoeff
from ready_db import ReadyDB
from lib import *
import swagger_client
from swagger_client import ApiClient
from swagger_client import Configuration
from swagger_client.rest import ApiException


model_path = os.path.expanduser("~/cp1/models/")
learned_model_path = os.path.expanduser("~/cp1/")
config_list_file = os.path.expanduser('~/cp1/config_list.json')
config_list_file_true = os.path.expanduser('~/cp1/config_list_true.json')
ready_json = os.path.expanduser("~/ready")
learned_model_name = 'learned_model'

ndim = 20
test_size = 10000
mu, sigma = 0, 0.1


ready = ReadyDB(ready_db=ready_json)
budget = ready.get_budget()
model_name = ready.get_power_model()

internal_api = swagger_client.DefaultApi()

default_conf = np.reshape(np.ones(ndim), (1, ndim))

try:
    with open(os.path.join(model_path, model_name), 'r') as model_file:
        model_txt = model_file.read()

    power_model_terms = genModelTermsfromString(model_txt)
    true_power_model = Model(power_model_terms, ndim)
    print("The true model: {0}".format(true_power_model.__str__()))
except Exception as e:
    internal = {
    "status": "parsing-error",
    "message": e.message
    }
    internal_api.internal_post(internal)


# xTrain = np.ones(ndim)
# xTrain = np.reshape(xTrain, (nrow, ndim))
# yTrain = power_model.evaluateModelFast(xTrain)
# print(yTrain)

print("Learning started")
internal = {
    "status": "learning-started",
    "message": "lets start learning the power model"
}
internal_api.internal_post(internal)


# learn the model
try:
    learner = MLearner(budget, ndim, true_power_model)
    learned_model = learner.discover()
except Exception as e:
    internal = {
    "status": "learning-error",
    "message": e.message
    }
    internal_api.internal_post(internal)


print("Learning done!")
internal = {
    "status": "learning-done",
    "message": "done with the learning"
    }
internal_api.internal_post(internal)

print(learned_model.named_steps['linear'].coef_)

learned_power_model_terms = genModelfromCoeff(learned_model.named_steps['linear'].coef_, ndim)
learned_power_model = Model(learned_power_model_terms, ndim)

print("The learned model: {0}".format(learned_power_model.__str__()))

with open(os.path.join(learned_model_path, learned_model_name), 'w') as model_file:
    model_file.write(learned_power_model.__str__())

# configs = itertools.product(range(2), repeat=ndim)
# xTest = np.zeros(shape=(2**ndim, ndim))
# i = 0
# for c in configs:
#     xTest[i, :] = np.array(c)
#     i += 1

xTest = np.random.randint(2, size=(test_size, ndim))
yTestPower = learned_model.predict(xTest)
yTestPower_true = true_power_model.evaluateModelFast(xTest)

# adding noise for the speed
s = np.random.normal(mu, sigma, test_size)
yTestSpeed = yTestPower_true/100 + s

yDefaultPower = learned_model.predict(default_conf)
yDefaultPower_true = true_power_model.evaluateModelFast(default_conf)
yDefaultSpeed = yDefaultPower_true/100

idx_pareto, pareto_power, pareto_speed = learner.get_pareto_frontier(yTestPower, yTestSpeed, maxX=False, maxY=True)
json_data = get_json(pareto_power, pareto_speed)

json_data_true_model = get_json([yTestPower_true[i] for i in idx_pareto], [yTestSpeed[i] for i in idx_pareto])

# add the default configuration
json_data['configurations'].append({
    'config_id': 0,
    'power_load': yDefaultPower[0]/3600*1000,
    'power_load_w': yDefaultPower[0],
    'speed': yDefaultSpeed[0]
})
with open(config_list_file, 'w') as outfile:
    json.dump(json_data, outfile)

json_data_true_model['configurations'].append({
    'config_id': 0,
    'power_load': yDefaultPower_true[0]/3600*1000,
    'power_load_w': yDefaultPower_true[0],
    'speed': yDefaultSpeed[0]
})
with open(config_list_file_true, 'w') as outfile:
    json.dump(json_data_true_model, outfile)


