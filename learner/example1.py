import mlearner

learner = mlearner.MLearner(10, 3, ("x", "y"), [[0,0],[5,10]], "func.xml")
learned_model = learner.discover()
coefficients = learned_model.named_steps['linear'].coef_
print(coefficients)
