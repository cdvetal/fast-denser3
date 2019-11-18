from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def accuracy(y_true, y_pred):
	"""
	    Computes the accuracy.


	    Parameters
	    ----------
	    y_true : np.array
	        array of right labels
	    
	    y_pred : np.array
	        array of class confidences for each instance
	    

	    Returns
	    -------
	    accuracy : float
	    	accuracy value
    """


	y_pred_labels = np.argmax(y_pred, axis=1)

	return accuracy_score(y_true, y_pred_labels)



def mse(y_true, y_pred):
	"""
	    Computes the mean squared error (MSE).


	    Parameters
	    ----------
	    y_true : np.array
	        array of right outputs
	    
	    y_pred : np.array
	        array of predicted outputs
	    

	    Returns
	    -------
	    mse : float
	    	mean squared errr
    """

	return mean_squred_error(y_true, y_pred)