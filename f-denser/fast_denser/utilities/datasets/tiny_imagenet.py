import numpy as np
from PIL import Image


def load_train(wnids, dataset_path, shape):
	"""
        Load the train files  dataset

        Parameters
        ----------
        dataset_path : str
        	path to the dataset files

        shape : tuple
        	target shape of the loaded instances

        Returns
        -------
        x_train : np.array
            training instances
        y_train : np.array
            training labels 
    """


	x_train = np.ndarray(shape = (100000, 32, 32, 3), dtype = np.uint8)
	y_train = np.ndarray(shape = (100000), dtype = np.uint8)
	for idx, wnid in enumerate(wnids):
		for j in range(500):
			im = Image.open('%s/train/%s/images/%s_%d.JPEG' % (dataset_path, wnid, wnid, j)).convert('RGB')

			if shape != (64, 64):
				im = im.resize((32, 32), Image.LANCZOS)

			x_train[idx*500+j] = np.asarray(im)
			y_train[idx*500+j] = idx

	return x_train, y_train



def load_test(wnids, dataset_path, shape):
	"""
        Load the test files  dataset

        Parameters
        ----------
        dataset_path : str
        	path to the dataset files

        shape : tuple
        	target shape of the loaded instances

        Returns
        -------
        x_test : np.array
            testing instances
        x_test : np.array
            testing labels
    """

	x_test = np.ndarray(shape = (10000, 32, 32, 3), dtype = np.uint8)
	y_test = np.ndarray(shape = (10000), dtype = np.uint8)

	for i, line in enumerate([s.strip() for s in open('%s/val/val_annotations.txt' % dataset_path)]):
		name, wnid = line.split('\t')[:2]
		
		im = Image.open('%s/val/images/%s' % (dataset_path, name)).convert('RGB')

		if shape != (64, 64):
			im = im.resize((32, 32), Image.LANCZOS)

		x_test[i] = np.asarray(im)
		y_test[i] = wnids.index(wnid)

	return x_test, y_test


def load_tiny_imagenet(dataset_path, shape):
	"""
        Load the tiny-imagenet dataset

        Parameters
        ----------
        dataset_path : str
        	path to the dataset files

        shape : tuple
        	target shape of the loaded instances

        Returns
        -------
        x_train : np.array
            training instances
        y_train : np.array
            training labels 
        x_test : np.array
            testing instances
        x_test : np.array
            testing labels
    """


	wnids = [x.strip() for x in open('%s/wnids.txt' % dataset_path).readlines()]

	x_train, y_train = load_train(wnids, dataset_path, shape)
	x_test, y_test = load_test(wnids, dataset_path, shape)

	return x_train, y_train, x_test, y_test


