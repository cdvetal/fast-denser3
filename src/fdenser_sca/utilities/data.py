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


from .datasets.svhn import load_svhn
from .datasets.cifar import load_cifar
from .datasets.tiny_imagenet import load_tiny_imagenet
from sklearn.model_selection import train_test_split
import keras
from multiprocessing import Pool
import tensorflow as tf
import contextlib
import sys


# dataset paths - change if the path is different
SVHN = 'fast_denser/utilities/datasets/data/svhn'
TINY_IMAGENET = 'fast_denser/utilities/datasets/data/tiny-imagenet-200'


def prepare_data(x_train, y_train, x_test, y_test, n_classes=10):
    """
        Split the data into independent sets

        Parameters
        ----------
        x_train : np.array
            training instances
        y_train : np.array
            training labels
        x_test : np.array
            testing instances
        x_test : np.array
            testing labels


        Returns
        -------
        dataset : dict
            instances of the dataset:
                For evolution:
                    - evo_x_train and evo_y_train : training x, and y instances
                    - evo_x_val and evo_y_val : validation x, and y instances
                                                used for early stopping
                    - evo_x_test and evo_y_test : testing x, and y instances
                                                  used for fitness assessment
                After evolution:
                    - x_test and y_test : for measusing the effectiveness of
                                          the model on unseen data
    """

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train.reshape((-1, 32, 32, 3))
    x_test = x_test.reshape((-1, 32, 32, 3))

    evo_x_train, x_val, evo_y_train, y_val = train_test_split(
        x_train, y_train, test_size=7000, stratify=y_train)

    evo_x_val, evo_x_test, evo_y_val, evo_y_test = train_test_split(
        x_val, y_val, test_size=3500, stratify=y_val)

    evo_y_train = keras.utils.to_categorical(evo_y_train, n_classes)
    evo_y_val = keras.utils.to_categorical(evo_y_val, n_classes)

    dataset = {
        'evo_x_train': evo_x_train, 'evo_y_train': evo_y_train,
        'evo_x_val': evo_x_val, 'evo_y_val': evo_y_val,
        'evo_x_test': evo_x_test, 'evo_y_test': evo_y_test,
        'x_test': x_test, 'y_test': y_test
    }

    return dataset


def resize_data(args):
    """
        Resize the dataset 28 x 28 datasets to 32x32

        Parameters
        ----------
        args : tuple(np.array, (int, int))
            instances, and shape of the reshaped signal

        Returns
        -------
        content : np.array
            reshaped instances
    """

    content, shape = args
    content = content.reshape(-1, 28, 28, 1)

    if shape != (28, 28):
        content = tf.image.resize(
            content, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    content = tf.image.grayscale_to_rgb(tf.constant(content))

    return content.numpy()


def load_dataset(dataset, shape=(32, 32)):
    """
        Load a specific dataset

        Parameters
        ----------
        dataset : str
            dataset to load

        shape : tuple(int, int)
            shape of the instances

        Returns
        -------
        dataset : dict
            instances of the dataset:
                For evolution:
                    - evo_x_train and evo_y_train : training x, and y instances
                    - evo_x_val and evo_y_val : validation x, and y instances
                                                used for early stopping
                    - evo_x_test and evo_y_test : testing x, and y instances
                                                  used for fitness assessment
                After evolution:
                    - x_test and y_test : for measusing the effectiveness of
                                          the model on unseen data
    """

    if dataset == 'fashion-mnist':
        (x_train, y_train), (x_test, y_test) = \
            keras.datasets.fashion_mnist.load_data()
        n_classes = 10

        x_train = 255-x_train
        x_test = 255-x_test

        num_pool_workers = 1
        with contextlib.closing(Pool(num_pool_workers)) as po:
            pool_results = po.map_async(resize_data, [(x_train, shape)])
            x_train = pool_results.get()[0]

        with contextlib.closing(Pool(num_pool_workers)) as po:
            pool_results = po.map_async(resize_data, [(x_test, shape)])
            x_test = pool_results.get()[0]

    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        n_classes = 10

        num_pool_workers = 1
        with contextlib.closing(Pool(num_pool_workers)) as po:
            pool_results = po.map_async(resize_data, [(x_train, shape)])
            x_train = pool_results.get()[0]

        with contextlib.closing(Pool(num_pool_workers)) as po:
            pool_results = po.map_async(resize_data, [(x_test, shape)])
            x_test = pool_results.get()[0]

    # 255, unbalanced
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn(SVHN)
        n_classes = 10

    # 255, 50000, 10000
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar(10)
        n_classes = 10

    # 255, 50000, 10000
    elif dataset == 'cifar100-fine':
        x_train, y_train, x_test, y_test = load_cifar(100, 'fine')
        n_classes = 100

    elif dataset == 'cifar100-coarse':
        x_train, y_train, x_test, y_test = load_cifar(100, 'coarse')
        n_classes = 20

    elif dataset == 'tiny-imagenet':
        x_train, y_train, x_test, y_test = \
            load_tiny_imagenet(TINY_IMAGENET, shape)
        n_classes = 200

    else:
        print('Error: the dataset is not valid')
        sys.exit(-1)

    dataset = prepare_data(x_train, y_train, x_test, y_test, n_classes)

    return dataset
