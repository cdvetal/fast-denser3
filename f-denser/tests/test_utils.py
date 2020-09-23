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

		print(ind.decode(grammar))

		return ind, grammar


	def test_pickle_evaluator(self):
		from fast_denser.utils import Evaluator
		from fast_denser.utilities.fitness_metrics import accuracy
		import fast_denser.engine as engine
		import os

		random.seed(0)
		ind, grammar = self.create_individual()
		evaluator = Evaluator('cifar10', accuracy)

		if not os.path.exists('./run_0/'):
			os.makedirs('./run_0/')

		engine.pickle_evaluator(evaluator, '.', 0)

		self.assertTrue(os.path.exists('./run_0/evaluator.pkl'))



	def test_save_population(self):
		from fast_denser.utils import Individual
		from fast_denser.grammar import Grammar
		import fast_denser.engine as engine
		import os
		import random

		network_structure = [["features", 1, 3]]
		grammar = Grammar('tests/utilities/example.grammar')
		levels_back = {"features": 1, "classification": 1}
		network_structure_init = {"features":[2]}

		ind = Individual(network_structure, [], 'softmax', 0)

		if not os.path.exists('./run_0/'):
			os.makedirs('./run_0/')

		engine.save_pop([ind], '.', 0, 0)

		self.assertTrue(os.path.exists('./run_0/gen_0.csv'))

		engine.pickle_population([ind], ind, '.', 0)

		self.assertTrue(os.path.exists('./run_0/population.pkl'))
		self.assertTrue(os.path.exists('./run_0/parent.pkl'))
		self.assertTrue(os.path.exists('./run_0/random.pkl'))
		self.assertTrue(os.path.exists('./run_0/numpy.pkl'))

		loaded_data = engine.unpickle_population('.', 0)

		self.assertTrue(loaded_data)


	def test_load_config(self):
		import fast_denser.engine as engine

		config = engine.load_config('tests/utilities/example_config.json')

		self.assertTrue(config)



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



if __name__ == '__main__':
    unittest.main()
