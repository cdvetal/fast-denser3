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


