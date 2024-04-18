import unittest
import warnings
import random

class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)


	def test_learning_mapping(self):
		from fast_denser.utils import Evaluator
		from fast_denser.utilities.fitness_metrics import accuracy

		random.seed(0)
		evaluator = Evaluator('mnist', accuracy)

		learning_params_rmsprop = evaluator.get_learning('learning:rmsprop lr:0.1 rho:1 decay:0.000001')
		optimiser_rmsprop = evaluator.assemble_optimiser(learning_params_rmsprop)
		self.assertTrue(optimiser_rmsprop, "Error assembling optimiser")

		learning_params_adam = evaluator.get_learning('learning:adam lr:0.1 beta1:0.5 beta2:0.5 decay:0.000001')
		optimiser_adam = evaluator.assemble_optimiser(learning_params_adam)
		self.assertTrue(optimiser_adam, "Error assembling optimiser")

		learning_params_gradient = evaluator.get_learning('learning:gradient-descent lr:0.1 momentum:0.68 decay:0.001 nesterov:True')
		optimiser_gradient = evaluator.assemble_optimiser(learning_params_gradient)
		self.assertTrue(optimiser_gradient, "Error assembling optimiser")


if __name__ == '__main__':
    unittest.main()
