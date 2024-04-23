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
from keras.callbacks import ModelCheckpoint
import os

from .timed_stopping import TimedStopping
from .utilities.data import load_dataset

# TODO: future -- impose memory constraints
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=50)],
# )

DEBUG = False


class Evaluator:
    """
        Stores the dataset, maps the phenotype into a trainable model, and
        evaluates it


        Attributes
        ----------
        dataset : dict
            dataset instances and partitions

        fitness_metric : function
            fitness_metric (y_true, y_pred)
            y_pred are the confidences


        Methods
        -------
        get_layers(phenotype)
            parses the phenotype corresponding to the layers
            auxiliary function of the assemble_network function

        get_learning(learning)
            parses the phenotype corresponding to the learning
            auxiliary function of the assemble_optimiser function

        assemble_network(keras_layers, input_size)
            maps the layers phenotype into a keras model

        assemble_optimiser(learning)
            maps the learning into a keras optimiser

        evaluate(phenotype, load_prev_weights, weights_save_path,
                 parent_weights_path, train_time, num_epochs, datagen=None,
                 input_size=(32, 32, 3))
            evaluates the keras model using the keras optimiser

        testing_performance(self, model_path)
            compute testing performance of the model
    """

    def __init__(self, dataset, fitness_metric):
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """

        # define the dataset on which the models will be evaluated
        self.dataset = load_dataset(dataset)
        # define the metric used to evaluate the models
        self.fitness_metric = fitness_metric

    def get_layers(self, phenotype):
        """
            Parses the phenotype corresponding to the layers.
            Auxiliary function of the assemble_network function.

            Parameters
            ----------
            phenotye : str
                individual layers phenotype

            Returns
            -------
            layers : list
                list of tuples (layer_type : str, node properties : dict)
        """

        raw_phenotype = phenotype.split(' ')

        idx = 0
        first = True
        node_type, node_val = raw_phenotype[idx].split(':')
        layers = []

        while idx < len(raw_phenotype):
            if node_type == 'layer':
                if not first:
                    layers.append((layer_type, node_properties))
                else:
                    first = False
                layer_type = node_val
                node_properties = {}
            else:
                node_properties[node_type] = node_val.split(',')

            idx += 1
            if idx < len(raw_phenotype):
                node_type, node_val = raw_phenotype[idx].split(':')

        layers.append((layer_type, node_properties))

        return layers

    def get_learning(self, learning):
        """
            Parses the phenotype corresponding to the learning
            Auxiliary function of the assemble_optimiser function

            Parameters
            ----------
            learning : str
                learning phenotype of the individual

            Returns
            -------
            learning_params : dict
                learning parameters
        """

        raw_learning = learning.split(' ')

        idx = 0
        learning_params = {}
        while idx < len(raw_learning):
            param_name, param_value = raw_learning[idx].split(':')
            learning_params[param_name] = param_value.split(',')
            idx += 1

        for _key_ in sorted(list(learning_params.keys())):
            if len(learning_params[_key_]) == 1:
                try:
                    learning_params[_key_] = eval(learning_params[_key_][0])
                except NameError:
                    learning_params[_key_] = learning_params[_key_][0]

        return learning_params

    def assemble_network(self, keras_layers, input_size):
        """
            Maps the layers phenotype into a keras model

            Parameters
            ----------
            keras_layers : list
                output from get_layers

            input_size : tuple
                network input shape

            Returns
            -------
            model : keras.models.Model
                keras trainable model
        """

        # input layer
        inputs = keras.layers.Input(shape=input_size)

        # Create layers -- ADD NEW LAYERS HERE
        layers = []
        for layer_type, layer_params in keras_layers:
            # convolutional layer
            if layer_type == 'conv':
                conv_layer = keras.layers.Conv2D(
                    filters=int(layer_params['num-filters'][0]),
                    kernel_size=(
                        int(layer_params['filter-shape'][0]),
                        int(layer_params['filter-shape'][0]),
                    ),
                    strides=(
                        int(layer_params['stride'][0]),
                        int(layer_params['stride'][0]),
                    ),
                    padding=layer_params['padding'][0],
                    activation=layer_params['act'][0],
                    use_bias=eval(layer_params['bias'][0]),
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.0005),
                )
                layers.append(conv_layer)

            # batch-normalisation
            elif layer_type == 'batch-norm':
                # TODO - check because channels are not first
                batch_norm = keras.layers.BatchNormalization()
                layers.append(batch_norm)

            # average pooling layer
            elif layer_type == 'pool-avg':
                pool_avg = keras.layers.AveragePooling2D(
                    pool_size=(
                        int(layer_params['kernel-size'][0]),
                        int(layer_params['kernel-size'][0]),
                    ),
                    strides=int(layer_params['stride'][0]),
                    padding=layer_params['padding'][0],
                )
                layers.append(pool_avg)

            # max pooling layer
            elif layer_type == 'pool-max':
                pool_max = keras.layers.MaxPooling2D(
                    pool_size=(
                        int(layer_params['kernel-size'][0]),
                        int(layer_params['kernel-size'][0]),
                    ),
                    strides=int(layer_params['stride'][0]),
                    padding=layer_params['padding'][0],
                )
                layers.append(pool_max)

            # fully-connected layer
            elif layer_type == 'fc':
                fc = keras.layers.Dense(
                    int(layer_params['num-units'][0]),
                    activation=layer_params['act'][0],
                    use_bias=eval(layer_params['bias'][0]),
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.0005),
                )
                layers.append(fc)

            # dropout layer
            elif layer_type == 'dropout':
                dropout = keras.layers.Dropout(
                    rate=min(0.5, float(layer_params['rate'][0])))
                layers.append(dropout)

            # gru layer #TODO: initializers, recurrent dropout, dropout,
            # unroll, reset_after
            elif layer_type == 'gru':
                gru = keras.layers.GRU(
                    units=int(layer_params['units'][0]),
                    activation=layer_params['act'][0],
                    recurrent_activation=layer_params['rec_act'][0],
                    use_bias=eval(layer_params['bias'][0]),
                )
                layers.append(gru)

            # lstm layer #TODO: initializers, recurrent dropout, dropout,
            # unroll, reset_after
            elif layer_type == 'lstm':
                lstm = keras.layers.LSTM(
                    units=int(layer_params['units'][0]),
                    activation=layer_params['act'][0],
                    recurrent_activation=layer_params['rec_act'][0],
                    use_bias=eval(layer_params['bias'][0]),
                )
                layers.append(lstm)

            # rnn #TODO: initializers, recurrent dropout, dropout, unroll,
            # reset_after
            elif layer_type == 'rnn':
                rnn = keras.layers.SimpleRNN(
                    units=int(layer_params['units'][0]),
                    activation=layer_params['act'][0],
                    use_bias=eval(layer_params['bias'][0]),
                )
                layers.append(rnn)

            elif layer_type == 'conv1d':    # todo initializer
                conv1d = keras.layers.Conv1D(
                    filters=int(layer_params['num-filters'][0]),
                    kernel_size=int(layer_params['kernel-size'][0]),
                    strides=int(layer_params['strides'][0]),
                    padding=layer_params['padding'][0],
                    activation=layer_params['activation'][0],
                    use_bias=eval(layer_params['bias'][0]),
                )
                layers.add(conv1d)

            # END ADD NEW LAYERS

        # Connection between layers
        for layer in keras_layers:
            layer[1]['input'] = list(map(int, layer[1]['input']))

        first_fc = True
        data_layers = []
        invalid_layers = []

        for layer_idx, layer in enumerate(layers):
            try:
                if len(keras_layers[layer_idx][1]['input']) == 1:
                    if keras_layers[layer_idx][1]['input'][0] == -1:
                        data_layers.append(layer(inputs))
                    else:
                        if keras_layers[layer_idx][0] == 'fc' and first_fc:
                            first_fc = False
                            flatten = keras.layers.Flatten()(
                                data_layers[keras_layers[layer_idx][1]['input'][0]])
                            data_layers.append(layer(flatten))
                            continue

                        data_layers.append(
                            layer(data_layers[keras_layers[layer_idx][1]['input'][0]]))

                else:
                    # Get minimum shape: when merging layers all the signals are
                    # converted to the minimum shape
                    minimum_shape = input_size[0]
                    for input_idx in keras_layers[layer_idx][1]['input']:
                        if input_idx != -1 and input_idx not in invalid_layers:
                            if data_layers[input_idx].shape[-3:][0] < minimum_shape:
                                minimum_shape = int(
                                    data_layers[input_idx].shape[-3:][0])

                    # Reshape signals to the same shape
                    merge_signals = []
                    for input_idx in keras_layers[layer_idx][1]['input']:
                        if input_idx == -1:
                            if inputs.shape[-3:][0] > minimum_shape:
                                actual_shape = int(inputs.shape[-3:][0])
                                merge_signals.append(
                                    keras.layers.MaxPooling2D(
                                        pool_size=(
                                            actual_shape-(minimum_shape-1),
                                            actual_shape-(minimum_shape-1),
                                        ),
                                        strides=1,
                                    )(inputs))
                            else:
                                merge_signals.append(inputs)

                        elif input_idx not in invalid_layers:
                            if data_layers[input_idx].shape[-3:][0] > minimum_shape:
                                actual_shape = int(
                                    data_layers[input_idx].shape[-3:][0]
                                )
                                merge_signals.append(
                                    keras.layers.MaxPooling2D(
                                        pool_size=(
                                            actual_shape-(minimum_shape-1),
                                            actual_shape-(minimum_shape-1),
                                        ),
                                        strides=1,
                                    )(data_layers[input_idx]))
                            else:
                                merge_signals.append(data_layers[input_idx])

                    if len(merge_signals) == 1:
                        merged_signal = merge_signals[0]
                    elif len(merge_signals) > 1:
                        merged_signal = keras.layers.concatenate(merge_signals)
                    else:
                        merged_signal = data_layers[-1]

                    data_layers.append(layer(merged_signal))
            except ValueError as e:
                data_layers.append(data_layers[-1])
                invalid_layers.append(layer_idx)
                if DEBUG:
                    print(keras_layers[layer_idx][0])
                    print(e)

        model = keras.models.Model(inputs=inputs, outputs=data_layers[-1])

        if DEBUG:
            model.summary()

        return model

    def assemble_optimiser(self, learning):
        """
            Maps the learning into a keras optimiser

            Parameters
            ----------
            learning : dict
                output of get_learning

            Returns
            -------
            optimiser : keras.optimizers.Optimizer
                keras optimiser that will be later used to train the model
        """

        initial_learning_rate = float(learning['lr'])
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate,
            decay_steps=1,
            decay_rate=float(learning['decay']),
        )

        if learning['learning'] == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=lr_schedule,
                                            rho=float(learning['rho']))

        elif learning['learning'] == 'gradient-descent':
            return keras.optimizers.SGD(learning_rate=lr_schedule,
                                        momentum=float(learning['momentum']),
                                        nesterov=bool(learning['nesterov']))

        elif learning['learning'] == 'adam':
            return keras.optimizers.Adam(learning_rate=lr_schedule,
                                         beta_1=float(learning['beta1']),
                                         beta_2=float(learning['beta2']))

    def evaluate(self, phenotype, load_prev_weights, weights_save_path,
                 parent_weights_path, train_time, num_epochs,
                 input_size=(32, 32, 3)):
        """
            Evaluates the keras model using the keras optimiser

            Parameters
            ----------
            phenotype : str
                individual phenotype

            load_prev_weights : bool
                resume training from a previous train or not

            weights_save_path : str
                path where to save the model weights after training

            parent_weights_path : str
                path to the weights of the previous training

            train_time : float
                maximum training time

            num_epochs : int
                maximum number of epochs

            datagen : keras.preprocessing.image.ImageDataGenerator
                Data augmentation method image data generator

            input_size : tuple
                dataset input shape


            Returns
            -------
            score_history : dict
                training data: loss and accuracy
        """

        model_phenotype, learning_phenotype = phenotype.split('learning:')
        learning_phenotype = 'learning:'+learning_phenotype.rstrip().lstrip()
        model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

        keras_layers = self.get_layers(model_phenotype)
        keras_learning = self.get_learning(learning_phenotype)
        batch_size = int(keras_learning['batch_size'])

        if load_prev_weights and os.path.exists(parent_weights_path):
            model = keras.models.load_model(parent_weights_path)

        else:
            if load_prev_weights:
                num_epochs = 0

            model = self.assemble_network(keras_layers, input_size)
            opt = self.assemble_optimiser(keras_learning)

            model.compile(optimizer=opt,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        # early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=int(keras_learning['early_stop']),
            restore_best_weights=True,
        )

        # time based stopping
        time_stop = TimedStopping(seconds=train_time, verbose=DEBUG)

        # save individual with the lowest validation loss
        # useful for when traaining is halted because of time
        monitor = ModelCheckpoint(weights_save_path, monitor='val_loss',
                                  verbose=DEBUG, save_best_only=True)

        trainable_count = model.count_params()

        score = model.fit(
            x=self.dataset['evo_x_train'],
            y=self.dataset['evo_y_train'],
            batch_size=batch_size,
            epochs=int(keras_learning['epochs']),
            steps_per_epoch=(self.dataset['evo_x_train'].shape[0]//batch_size),
            validation_data=(self.dataset['evo_x_val'],
                             self.dataset['evo_y_val']),
            callbacks=[early_stop, time_stop, monitor],
            initial_epoch=num_epochs,
            verbose=DEBUG
        )

        # save final moodel to file
        model.save(weights_save_path)

        # measure test performance
        y_pred_test = model.predict(
            self.dataset['evo_x_test'], batch_size=batch_size, verbose=0)

        accuracy_test = self.fitness_metric(
            self.dataset['evo_y_test'], y_pred_test)

        if DEBUG:
            print(phenotype, accuracy_test)

        score.history['trainable_parameters'] = trainable_count
        score.history['accuracy_test'] = accuracy_test

        keras.backend.clear_session()

        return score.history

    def testing_performance(self, model_path):
        """
            Compute testing performance of the model

            Parameters
            ----------
            model_path : str
                Path to the model .h5 file


            Returns
            -------
            accuracy : float
                Model accuracy
        """

        model = keras.models.load_model(model_path)
        y_pred = model.predict(self.dataset['x_test'])

        accuracy = self.fitness_metric(self.dataset['y_test'], y_pred)
        return accuracy
