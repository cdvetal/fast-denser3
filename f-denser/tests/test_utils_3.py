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

		layers = evaluator.get_layers('layer:batch-norm input:-1 layer:pool-avg kernel-size:2 stride:1 input:0 padding:same layer:pool-max kernel-size:2 input:1 stride:1 padding:same layer:dropout rate:0.5 input:2 layer:fc num-units:10 input:2 act:relu bias:True')
		network = evaluator.assemble_network(layers, input_size=(32, 32, 3))

		self.assertTrue(network, "Error network was not created")
		


if __name__ == '__main__':
    unittest.main()
