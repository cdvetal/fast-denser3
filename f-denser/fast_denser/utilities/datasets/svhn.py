import scipy.io

def load_mat(path):
	"""
        Load SVHN mat files

        Parameters
        ----------
        dataset_path : str
        	path to the dataset files

        Returns
        -------
        x : np.array
            instances
        y : np.array
			labels
    """

	data = scipy.io.loadmat(path)
	x = data['X']
	y = data['y']-1

	x = np.rollaxis(x, 3, 0)
	y = y.reshape(-1)

	return x, y

def load_svhn(dataset_path):
	"""
        Load the SVHN dataset

        Parameters
        ----------
        dataset_path : str
        	path to the dataset files

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


	x_train, y_train = load_mat('%s/train_32x32.mat' % dataset_path)
	x_test, y_test = load_mat('%s/test_32x32.mat' % dataset_path)

	return x_train, y_train, x_test, y_test


