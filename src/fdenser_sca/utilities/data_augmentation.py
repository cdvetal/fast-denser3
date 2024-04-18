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


import numpy as np

def augmentation(x):
    """
        Data augmentation strategy used for training the networks:
            . padding;
            . cropping;
            . horizontal flipping.

        Parameters
        ----------
        x : numpy.array (of rank 3)
            input image


        Returns
        -------
        aug_data : numpy.array
            augmented version of the image

    """

    pad_size = 4

    h, w, c = 32, 32, 3
    pad_h = h + 2 * pad_size
    pad_w = w + 2 * pad_size

    #Padding
    pad_img = np.zeros((pad_h, pad_w, c))
    pad_img[pad_size:h+pad_size, pad_size:w+pad_size, :] = x

    #Cropping and horizontal flip the image
    top = np.random.randint(0, pad_h - h + 1)
    left = np.random.randint(0, pad_w - w + 1)
    bottom = top + h
    right = left + w
    if np.random.randint(0, 2):
        pad_img = pad_img[:, ::-1, :]

    aug_data = pad_img[top:bottom, left:right, :]

    return aug_data

