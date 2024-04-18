# Copyright 2019 Filipe Assuncao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

	return mean_squared_error(y_true, y_pred)