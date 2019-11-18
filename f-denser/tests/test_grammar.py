import unittest
import warnings

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

		self.assertEqual(str(grammar), output, "Error: grammars differ")


	def test_decode(self):
		import fast_denser.grammar

		grammar = fast_denser.grammar.Grammar('tests/utilities/example.grammar')

		start_symbol = 'features'
		genotype = {'padding': [{'ge': 1, 'ga': {}}], 'bias': [{'ge': 0, 'ga': {}}], 'features': [{'ge': 0, 'ga': {}}], 'activation-function': [{'ge': 2, 'ga': {}}], 'convolution': [{'ge': 0, 'ga': {'num-filters': ('int', 32.0, 256.0, [242]), 'filter-shape': ('int', 2.0, 5.0, [5]), 'stride': ('int', 1.0, 3.0, [2])}}]}
		output = "layer:conv num-filters:242 filter-shape:5 stride:2 padding:valid act:sigmoid bias:True"

		phenotype = grammar.decode(start_symbol, genotype)


		self.assertEqual(phenotype, output, "Error: phenotypes differ")





if __name__ == '__main__':
    unittest.main()