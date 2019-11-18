import keras

def load_cifar(n_classes=10, label_type='fine'):
	"""
        Load the cifar dataset

        Parameters
        ----------
        n_classes : int
        	number of problem classes

        label_type : str
        	label type of the cifar100 dataset


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

	if n_classes == 10:
		(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
	elif n_classes == 100:
		(x_train , y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode=label_type)


	return x_train, y_train, x_test, y_test


