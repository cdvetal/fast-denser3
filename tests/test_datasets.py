import unittest
import warnings

class Test(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter('ignore', category=DeprecationWarning)


	def test_load_datasets(self):
		import fast_denser.utilities.data as data

		fashion_mnist = data.load_dataset(dataset='fashion-mnist')
		mnist = data.load_dataset(dataset='mnist')
		svhn = data.load_dataset(dataset='svhn')
		cifar_10 = data.load_dataset(dataset='cifar10')
		cifar_100_fine = data.load_dataset(dataset='cifar100-fine')
		cifar_100_coarse = data.load_dataset(dataset='cifar100-coarse')
		tiny_imagenet = data.load_dataset(dataset='tiny-imagenet')

		self.assertTrue(fashion_mnist, "Error loading fashion-mnist")
		self.assertTrue(mnist, "Error loading mnist")
		self.assertTrue(svhn, "Error loading svhn")
		self.assertTrue(cifar_10, "Error loading cifar-10")
		self.assertTrue(cifar_100_fine, "Error loading cifar-100-fine")
		self.assertTrue(cifar_100_coarse, "Error loading cifar-100-coarse")
		self.assertTrue(tiny_imagenet, "Error loading cifar-100-coarse")

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


	def test_resize_data(self):
		import fast_denser.utilities.data as data
		import numpy as np

		data_resized = data.resize_data((np.zeros(shape=(10,28,28,1)), (28, 28)))

		self.assertEqual(data_resized.shape[1:], (28, 28, 3), "Error resizing data")



if __name__ == '__main__':
    unittest.main()