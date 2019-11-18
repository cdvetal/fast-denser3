import unittest
import warnings
import random

class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)


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
		from fast_denser.f_denser import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 1, 0, 0, 0, 0, 0, 0, 0, 60)

		self.assertEqual(self.count_unique_layers(ind.modules)+1, self.count_unique_layers(new_ind.modules), "Error: add layer wrong size")
		self.assertEqual(self.count_layers(ind.modules)+1, self.count_layers(new_ind.modules), "Error: add layer wrong size")


	def test_add_layer_replicate(self):
		from fast_denser.f_denser import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 1, 1, 0, 0, 0, 0, 0, 0, 60)

		self.assertEqual(self.count_unique_layers(ind.modules), self.count_unique_layers(new_ind.modules), "Error: duplicate layer wrong size")
		self.assertEqual(self.count_layers(ind.modules)+1, self.count_layers(new_ind.modules), "Error: duplicate layer wrong size")


	def test_remove_layer(self):
		from fast_denser.f_denser import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 0, 0, 1, 0, 0, 0, 0, 0, 60)

		self.assertEqual(self.count_layers(ind.modules)-1, self.count_layers(new_ind.modules), "Error: remove layer wrong size")
	
	def test_mutate_ge(self):
		from fast_denser.f_denser import mutation

		random.seed(0)
		ind, grammar = self.create_individual()
		
		num_layers_before_mutation = len(ind.modules[0].layers)

		new_ind = mutation(ind, grammar, 0, 0, 0, 0, 0, 1, 0, 0, 60)

		self.assertEqual(self.count_layers(ind.modules), self.count_layers(new_ind.modules), "Error: change ge parameter")

		count_differences = 0
		for module_idx in range(len(ind.modules)):
			for layer_idx in range(len(ind.modules[module_idx].layers)):
				if ind.modules[module_idx].layers[layer_idx] != new_ind.modules[module_idx].layers[layer_idx]:
					count_differences += 1

		self.assertEqual(1, count_differences, "Error: change ge parameter")

	
	def test_keras_mapping(self):
		from fast_denser.utils import Evaluator
		from fast_denser.utilities.fitness_metrics import accuracy

		random.seed(0)
		ind, grammar = self.create_individual()
		evaluator = Evaluator('mnist', accuracy)

		phenotype = ind.decode(grammar)
		keras_layers = evaluator.get_layers(phenotype)
		model = evaluator.assemble_network(keras_layers, (32, 32, 3))

		expected_output = {'layers': [{'class_name': 'InputLayer', 'config': {'dtype': 'float32', 'batch_input_shape': (None, 32, 32, 3), 'name': 'input_1', 'sparse': False}, 'inbound_nodes': [], 'name': 'input_1'}, {'class_name': 'Conv2D', 'config': {'kernel_constraint': None, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'distribution': 'normal', 'scale': 2.0, 'seed': None, 'mode': 'fan_in'}}, 'name': 'conv2d_1', 'bias_regularizer': None, 'bias_constraint': None, 'activation': 'linear', 'trainable': True, 'data_format': 'channels_last', 'padding': 'valid', 'strides': (2, 2), 'dilation_rate': (1, 1), 'kernel_regularizer': {'class_name': 'L1L2', 'config': {'l2': 0.0005000000237487257, 'l1': 0.0}}, 'filters': 90, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'use_bias': True, 'activity_regularizer': None, 'kernel_size': (4, 4)}, 'inbound_nodes': [[['input_1', 0, 0, {}]]], 'name': 'conv2d_1'}, {'class_name': 'Conv2D', 'config': {'kernel_constraint': None, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'distribution': 'normal', 'scale': 2.0, 'seed': None, 'mode': 'fan_in'}}, 'name': 'conv2d_2', 'bias_regularizer': None, 'bias_constraint': None, 'activation': 'sigmoid', 'trainable': True, 'data_format': 'channels_last', 'padding': 'same', 'strides': (2, 2), 'dilation_rate': (1, 1), 'kernel_regularizer': {'class_name': 'L1L2', 'config': {'l2': 0.0005000000237487257, 'l1': 0.0}}, 'filters': 95, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'use_bias': False, 'activity_regularizer': None, 'kernel_size': (5, 5)}, 'inbound_nodes': [[['conv2d_1', 0, 0, {}]]], 'name': 'conv2d_2'}, {'class_name': 'Flatten', 'config': {'trainable': True, 'name': 'flatten_1', 'data_format': 'channels_last'}, 'inbound_nodes': [[['conv2d_2', 0, 0, {}]]], 'name': 'flatten_1'}, {'class_name': 'Dense', 'config': {'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'distribution': 'normal', 'scale': 2.0, 'seed': None, 'mode': 'fan_in'}}, 'name': 'dense_1', 'kernel_constraint': None, 'bias_regularizer': None, 'bias_constraint': None, 'activation': 'softmax', 'trainable': True, 'kernel_regularizer': {'class_name': 'L1L2', 'config': {'l2': 0.0005000000237487257, 'l1': 0.0}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'units': 10, 'use_bias': True, 'activity_regularizer': None}, 'inbound_nodes': [[['flatten_1', 0, 0, {}]]], 'name': 'dense_1'}], 'input_layers': [['input_1', 0, 0]], 'output_layers': [['dense_1', 0, 0]], 'name': 'model_1'}

		self.assertEqual(expected_output, model.get_config(), "Error: mapping")


if __name__ == '__main__':
    unittest.main()