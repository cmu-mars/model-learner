import numpy as np
import os
import json

# the main for learning

from learner.mlearner import MLearner
from learner.model import genModelTermsfromString, Model, genModelfromCoeff
from learner.ready_db import ReadyDB
from learner.lib import *

model_path = os.path.expanduser("~/catkin_ws/src/cp1_base/power_models/")
learned_model_path = os.path.expanduser("~/cp1/")
config_list_file = os.path.expanduser('~/cp1/config_list.json')
config_list_file_true = os.path.expanduser('~/cp1/config_list_true.json')
ready_json = os.path.expanduser("~/ready")
learned_model_name = 'learned_model'

ndim = 20
test_size = 10000
mu, sigma = 0, 0.1
speed_list = [0.17, 0.34, 0.68]


class Learn:
    def __init__(self):
        self.ready = ReadyDB(ready_db=ready_json)
        self.budget = self.ready.get_budget()
        self.model_name = self.ready.get_power_model()
        self.default_conf = np.reshape(np.ones(ndim), (1, ndim))
        self.true_power_model = None
        self.learned_power_model = None
        self.learned_model = None
        self.learner = None

    def get_true_model(self):
        try:
            with open(os.path.join(model_path, self.model_name), 'r') as model_file:
                model_txt = model_file.read()

            power_model_terms = genModelTermsfromString(model_txt)
            self.true_power_model = Model(power_model_terms, ndim)
            print("The true model: {0}".format(self.true_power_model.__str__()))
            return self.true_power_model
        except Exception as e:
            raise Exception(e)

    def start_learning(self):

        # learn the model
        try:
            self.learner = MLearner(self.budget, ndim, self.true_power_model)
            self.learned_model = self.learner.discover()
        except Exception as e:
            raise Exception(e)

    def dump_learned_model(self):
        """dumps model in ~/cp1/"""

        try:
            learned_power_model_terms = genModelfromCoeff(self.learned_model.named_steps['linear'].coef_, ndim)
            self.learned_power_model = Model(learned_power_model_terms, ndim)
        except Exception as e:
            raise Exception(e)

        print("The learned model: {0}".format(self.learned_power_model.__str__()))

        with open(os.path.join(learned_model_path, learned_model_name), 'w') as model_file:
            model_file.write(self.learned_power_model.__str__())

        # configs = itertools.product(range(2), repeat=ndim)
        # xTest = np.zeros(shape=(2**ndim, ndim))
        # i = 0
        # for c in configs:
        #     xTest[i, :] = np.array(c)
        #     i += 1

        xTest = np.random.randint(2, size=(test_size, ndim))
        yTestPower = self.learned_model.predict(xTest)
        yTestPower_true = self.true_power_model.evaluateModelFast(xTest)

        # adding noise for the speed
        # s = np.random.normal(mu, sigma, test_size)

        yTestSpeed = []
        for i in range(test_size):
            yTestSpeed[i] = speed_list[i % len(speed_list)]

        yDefaultPower = self.learned_model.predict(self.default_conf)
        yDefaultPower_true = self.true_power_model.evaluateModelFast(self.default_conf)
        yDefaultSpeed = speed_list[2]

        idx_pareto, pareto_power, pareto_speed = self.learner.get_pareto_frontier(yTestPower, yTestSpeed, maxX=False, maxY=True)
        json_data = get_json(pareto_power, pareto_speed)

        json_data_true_model = get_json([yTestPower_true[i] for i in idx_pareto], [yTestSpeed[i] for i in idx_pareto])

        # add the default configuration
        json_data['configurations'].append({
            'config_id': 0,
            'power_load': yDefaultPower[0]/3600*1000,
            'power_load_w': yDefaultPower[0],
            'speed': yDefaultSpeed
        })
        with open(config_list_file, 'w') as outfile:
            json.dump(json_data, outfile)

        json_data_true_model['configurations'].append({
            'config_id': 0,
            'power_load': yDefaultPower_true[0]/3600*1000,
            'power_load_w': yDefaultPower_true[0],
            'speed': yDefaultSpeed
        })
        with open(config_list_file_true, 'w') as outfile:
            json.dump(json_data_true_model, outfile)


