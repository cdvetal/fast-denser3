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


import scipy.io
import numpy as np
import sys

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

    try:
       x_train, y_train = load_mat('%s/train_32x32.mat' % dataset_path)
       x_test, y_test = load_mat('%s/test_32x32.mat' % dataset_path)
    except FileNotFoundError:
       print("Error: you need to download the SVHN files first.")
       sys.exit(-1)

    return x_train, y_train, x_test, y_test


