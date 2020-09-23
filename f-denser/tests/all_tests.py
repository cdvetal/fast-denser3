import unittest
import warnings
import random

class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)


	def test_read_grammar(self):
		import fast_denser.grammar

		grammar = fast_denser.grammar.Grammar('tests/utilities/example.grammar')
		output = """<activation-function> ::= act:linear |  act:relu |  act:sigmoid
<bias> ::= bias:True |  bias:False
<convolution> ::= layer:conv [num-filters,int,1,32,256] [filter-shape,int,1,2,5] [stride,int,1,1,3] <padding> <activation-function> <bias>
<features> ::= <convolution>
<padding> ::= padding:same |  padding:valid
<softmax> ::= layer:fc act:softmax num-units:10 bias:True
"""
		grammar._str_()
		self.assertEqual(grammar.__str__(), output, "Error: grammars differ")


	def test_read_invalid_grammar(self):
		import fast_denser.grammar

		with self.assertRaises(SystemExit) as cm:
			grammar = fast_denser.grammar.Grammar('invalid_path')
			self.assertEqual(cm.exception.code, -1, "Error: read invalid grammar")


	def test_initialise(self):
		import fast_denser.grammar
		import random
		import numpy as np

		random.seed(0)
		np.random.seed(0)

		output = {'features': [{'ge': 0, 'ga': {}}], 'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 32.0, 256.0, [42]), 'filter-shape': ('int', 2.0, 5.0, [4]), 'stride': ('int', 1.0, 3.0, [3])}}], 'padding': [{'ge': 1, 'ga': {}}], 'activation-function': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 1, 'ga': {}}]}

		grammar = fast_denser.grammar.Grammar('tests/utilities/example.grammar')
		
		self.assertEqual(grammar.initialise('features'), output, "Error: initialise not equal")



	def test_decode(self):
		import fast_denser.grammar

		grammar = fast_denser.grammar.Grammar('tests/utilities/example.grammar')

		start_symbol = 'features'
		genotype = {'padding': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'features': [{'ge': 0, 'ga': {}}], 'activation-function': [{'ge': 2, 'ga': {}}], 'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 32.0, 256.0, [242]), 'filter-shape': ('int', 2.0, 5.0, [5]), 'stride': ('int', 1.0, 3.0, [2])}}]}
		output = "layer:conv num-filters:242 filter-shape:5 stride:2 padding:valid act:sigmoid bias:True"

		phenotype = grammar.decode(start_symbol, genotype)


		self.assertEqual(phenotype, output, "Error: phenotypes differ")


	def test_load_datasets(self):
		import fast_denser.utilities.data as data

	# 	fashion_mnist = data.load_dataset(dataset='fashion-mnist')
	# 	mnist = data.load_dataset(dataset='mnist')
	# 	svhn = data.load_dataset(dataset='svhn')
	# 	cifar_10 = data.load_dataset(dataset='cifar10')
	# 	cifar_100_fine = data.load_dataset(dataset='cifar100-fine')
	# 	cifar_100_coarse = data.load_dataset(dataset='cifar100-coarse')

	# 	self.assertTrue(fashion_mnist, "Error loading fashion-mnist")
	# 	self.assertTrue(mnist, "Error loading mnist")
	# 	self.assertTrue(svhn, "Error loading svhn")
	# 	self.assertTrue(cifar_10, "Error loading cifar-10")
	# 	self.assertTrue(cifar_100_fine, "Error loading cifar-100-fine")
	# 	self.assertTrue(cifar_100_coarse, "Error loading cifar-100-coarse")

		with self.assertRaises(SystemExit) as cm:
			other = data.load_dataset(dataset='not valid')
			self.assertEqual(cm.exception.code, -1, "Error: read invalid grammar")


	def test_augmentation(self):
		import fast_denser.utilities.data as data
		import fast_denser.utilities.data_augmentation as data_augmentation
		import tensorflow as tf

		cifar_10 = data.load_dataset(dataset='cifar10')
		input_image = cifar_10['evo_x_train'][0]
		augmented_image = data_augmentation.augmentation(input_image)
		diff = input_image - augmented_image

		self.assertTrue(diff.sum() != 0, "Error augmenting an image")


	def test_accuracy(self):
		import fast_denser.utilities.fitness_metrics as fitness

		self.assertEqual(fitness.accuracy([0, 1], [[0, 1], [0.5, 1]]), 0.5, "Error: accuracy is wrorng")


	def test_mse(self):
		import fast_denser.utilities.fitness_metrics as fitness

		self.assertEqual(fitness.mse([0, 1], [1, 0]), 1, "Error: mse is wrorng")

	def count_unique_layers(self, modules):
		unique_layers = []
		for module in modules:
			for layer in module.layers:
				unique_layers.append(id(layer))
		
		return len(set(unique_layers))

	def count_layers(self, modules):
		return sum([len(module.layers) for module in modules])

	def create_individual(self):
		from fast_denser.utils import Individual
		from fast_denser.grammar import Grammar

		network_structure = [["features", 1, 3]]
		grammar = Grammar('tests/utilities/example.grammar')
		levels_back = {"features": 1, "classification": 1}
		network_structure_init = {"features":[2]}

		ind = Individual(network_structure, [], 'softmax', 0).initialise(grammar, levels_back, 0, network_structure_init)

		return ind, grammar

	def test_add_layer_random(self):
		from fast_denser.engine import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 1, 0, 0, 0, 0, 0, 0, 0, 60)

		self.assertEqual(self.count_unique_layers(ind.modules)+1, self.count_unique_layers(new_ind.modules), "Error: add layer wrong size")
		self.assertEqual(self.count_layers(ind.modules)+1, self.count_layers(new_ind.modules), "Error: add layer wrong size")


	def test_add_layer_replicate(self):
		from fast_denser.engine import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 1, 1, 0, 0, 0, 0, 0, 0, 60)

		self.assertEqual(self.count_unique_layers(ind.modules), self.count_unique_layers(new_ind.modules), "Error: duplicate layer wrong size")
		self.assertEqual(self.count_layers(ind.modules)+1, self.count_layers(new_ind.modules), "Error: duplicate layer wrong size")


	def test_remove_layer(self):
		from fast_denser.engine import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 0, 0, 1, 0, 0, 0, 0, 0, 60)

		self.assertEqual(self.count_layers(ind.modules)-1, self.count_layers(new_ind.modules), "Error: remove layer wrong size")
	
	def test_mutate_ge(self):
		from fast_denser.engine import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 0, 0, 0, 0, 0, 1, 0, 0, 60)

		self.assertEqual(self.count_layers(ind.modules), self.count_layers(new_ind.modules), "Error: change ge parameter")

		count_ref = list()
		count_differences = 0
		total_dif = 0
		for module_idx in range(len(ind.modules)):
			for layer_idx in range(len(ind.modules[module_idx].layers)):
				total_dif += 1
				if ind.modules[module_idx].layers[layer_idx] != new_ind.modules[module_idx].layers[layer_idx]:
					if id(ind.modules[module_idx].layers[layer_idx]) not in count_ref:
						count_ref.append(id(ind.modules[module_idx].layers[layer_idx]))
						count_differences += 1

		self.assertEqual(total_dif, count_differences, "Error: change ge parameter")

	
	def test_keras_mapping(self):
		from fast_denser.utils import Evaluator
		from fast_denser.utilities.fitness_metrics import accuracy

		random.seed(0)
		ind, grammar = self.create_individual()
		evaluator = Evaluator('mnist', accuracy)

		phenotype = ind.decode(grammar)
		keras_layers = evaluator.get_layers(phenotype)
		model = evaluator.assemble_network(keras_layers, (32, 32, 3))

		expected_output = {'name': 'model_1', 'layers': [{'name': 'input_1', 'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, 32, 32, 3), 'dtype': 'float32', 'sparse': False, 'name': 'input_1'}, 'inbound_nodes': []}, {'name': 'conv2d_1', 'class_name': 'Conv2D', 'config': {'name': 'conv2d_1', 'trainable': True, 'dtype': 'float32', 'filters': 98, 'kernel_size': (5, 5), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'activation': 'relu', 'use_bias': False, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 2.0, 'mode': 'fan_in', 'distribution': 'normal', 'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': {'class_name': 'L1L2', 'config': {'l1': 0.0, 'l2': 0.0005000000237487257}}, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'inbound_nodes': [[['input_1', 0, 0, {}]]]}, {'name': 'conv2d_2', 'class_name': 'Conv2D', 'config': {'name': 'conv2d_2', 'trainable': True, 'dtype': 'float32', 'filters': 104, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'valid', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'activation': 'sigmoid', 'use_bias': True, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 2.0, 'mode': 'fan_in', 'distribution': 'normal', 'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': {'class_name': 'L1L2', 'config': {'l1': 0.0, 'l2': 0.0005000000237487257}}, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'inbound_nodes': [[['conv2d_1', 0, 0, {}]]]}, {'name': 'flatten_1', 'class_name': 'Flatten', 'config': {'name': 'flatten_1', 'trainable': True, 'dtype': 'float32', 'data_format': 'channels_last'}, 'inbound_nodes': [[['conv2d_2', 0, 0, {}]]]}, {'name': 'dense_1', 'class_name': 'Dense', 'config': {'name': 'dense_1', 'trainable': True, 'dtype': 'float32', 'units': 10, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 2.0, 'mode': 'fan_in', 'distribution': 'normal', 'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': {'class_name': 'L1L2', 'config': {'l1': 0.0, 'l2': 0.0005000000237487257}}, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'inbound_nodes': [[['flatten_1', 0, 0, {}]]]}], 'input_layers': [['input_1', 0, 0]], 'output_layers': [['dense_1', 0, 0]]}

		self.assertEqual(expected_output, model.get_config(), "Error: mapping")


	def test_learning_mapping(self):
		from fast_denser.utils import Evaluator
		from fast_denser.utilities.fitness_metrics import accuracy

		random.seed(0)
		evaluator = Evaluator('mnist', accuracy)

		learning_params_rmsprop = evaluator.get_learning('learning:rmsprop lr:0.1 rho:1 decay:0.000001')
		optimiser_rmsprop = evaluator.assemble_optimiser(learning_params)
		self.assertEqual(optimiser_rmsprop, "Error assembling optimiser")

		learning_params_adam = evaluator.get_learning('learning:adam lr:0.1 beta1:0.5 beta2:0.5 decay:0.000001')
		optimiser_adam = evaluator.assemble_optimiser(learning_params)
		self.assertEqual(optimiser_adam, "Error assembling optimiser")

		learning_params_gradient = evaluator.get_learning('learning:gradient-descent lr:0.1 momentum:0.68 decay:0.001 nesterov:True')
		optimiser_gradient = evaluator.assemble_optimiser(optimiser_gradient)
		self.assertEqual(optimiser_gradient, "Error assembling optimiser")

		


if __name__ == '__main__':
    unittest.main()