# -*- coding: utf-8 -*-
from polyparser.func_util import function_utilities
import unittest
from learner import mlearner

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_absolute_truth_and_meaning(self):
        assert True

    def test_learner(self):
        expwriter = function_utilities.Writer("func.xml")
        expwriter.writeexpr("x^3 + y")
        learner = mlearner.MLearner(10, 3, ("x", "y"), [[0, 0], [5, 10]], "func.xml")
        learned_model = learner.discover()
        coefficients = learned_model.named_steps['linear'].coef_
        # [1 x y x^2 xy y^2 x^3 x^2y xy^2 y^3] - > idx 2 and 6
        self.assertTrue((coefficients[0,2] >= 0.9) & (coefficients[0,6] >= 0.9))

    def test_learner1(self):
        expwriter = function_utilities.Writer("func.xml")
        expwriter.writeexpr("x + y")
        learner = mlearner.MLearner(10, 3, ("x", "y"), [[0, 0], [5, 10]], "func.xml")
        learned_model = learner.discover()
        coefficients = learned_model.named_steps['linear'].coef_
        # [1 x y x^2 xy y^2 x^3 x^2y xy^2 y^3] - > idx 2 and 6
        self.assertTrue((coefficients[0,1] >= 0.9) & (coefficients[0,2] >= 0.9))

    def test_learner2(self):
        expwriter = function_utilities.Writer("func.xml")
        expwriter.writeexpr("x1 + x2")
        learner = mlearner.MLearner(10, 3, ("x1", "x2"), [[0, 0], [5, 10]], "func.xml")
        learned_model = learner.discover()
        coefficients = learned_model.named_steps['linear'].coef_
        # [1 x y x^2 xy y^2 x^3 x^2y xy^2 y^3] - > idx 2 and 6
        self.assertTrue((coefficients[0,1] >= 0.9) & (coefficients[0,2] >= 0.9))



# if __name__ == '__main__':
#     unittest.main()

suite = unittest.TestLoader().loadTestsFromTestCase(BasicTestSuite)
unittest.TextTestRunner(verbosity=2).run(suite)