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


import os
import pickle
import random
from copy import deepcopy
from os import makedirs
from pathlib import Path
from shutil import copyfile

import yaml
import numpy as np

from .evaluator import Evaluator
from .execution import (
    pickle_evaluator,
    pickle_population,
    save_pop,
    unpickle_population,
)
from .grammar import Grammar
from .individual import Individual
from .utilities import fitness_metrics

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def select_fittest(population, population_fits, grammar, cnn_eval, gen,
                   run_path, default_train_time):
    """
        Select the parent to seed the next generation.


        Parameters
        ----------
        population : list
            list of instances of Individual

        population_fits : list
            ordered list of fitnesses of the population of individuals

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the
            genotype to phenotype mapping

        cnn_eval : Evaluator
            Evaluator instance used to train the networks

        datagen : keras.preprocessing.image.ImageDataGenerator
            Data augmentation method image data generator for the training data

        datagen_test : keras.preprocessing.image.ImageDataGenerator
            Data augmentation method image data generator for the validation
            and test data

        gen : int
            current generation of the ES

        save_path: str
            path where the ojects needed to resume evolution are stored.

        default_train_time : int
            default training time


        Returns
        -------
        parent : Individual
            individual that seeds the next generation
    """

    # Get best individual just according to fitness
    idx_max = np.argmax(population_fits)
    parent = population[idx_max]

    # however if the parent is not the elite, and the parent is trained for
    # longer, the elite is granted the same evaluation time.
    if parent.train_time > default_train_time:
        retrain_elite = False
        if idx_max != 0 and \
           population[0].train_time > default_train_time and \
           population[0].train_time < parent.train_time:
            retrain_elite = True
            elite = population[0]
            elite.train_time = parent.train_time
            elite.evaluate(
                grammar,
                cnn_eval,
                '%s/best_%d_%d.hdf5' % (run_path, gen, elite.id),
                '%s/best_%d_%d.hdf5' % (run_path, gen, elite.id),
            )
            population_fits[0] = elite.fitness

        min_train_time = min([ind.current_time for ind in population])

        # also retrain the best individual that is trained just for the
        # default time
        retrain_10min = False
        if min_train_time < parent.train_time:
            ids_10min = [ind.current_time == min_train_time
                         for ind in population]

            if sum(ids_10min) > 0:
                retrain_10min = True
                indvs_10min = np.array(population)[ids_10min]
                max_fitness_10min = max([ind.fitness for ind in indvs_10min])
                idx_max_10min = np.argmax(max_fitness_10min)
                parent_10min = indvs_10min[idx_max_10min]

                parent_10min.train_time = parent.train_time

                parent_10min.evaluate(
                    grammar,
                    cnn_eval,
                    '%s/best_%d_%d.hdf5' % (run_path, gen, parent_10min.id),
                    '%s/best_%d_%d.hdf5' % (run_path, gen, parent_10min.id),
                )

                population_fits[population.index(parent_10min)] = parent_10min.fitness

        # select the fittest amont all retrains and the initial parent
        if retrain_elite:
            if retrain_10min:
                if parent_10min.fitness > elite.fitness and \
                        parent_10min.fitness > parent.fitness:
                    return deepcopy(parent_10min)
                elif elite.fitness > parent_10min.fitness and \
                        elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
            else:
                if elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
        elif retrain_10min:
            if parent_10min.fitness > parent.fitness:
                return deepcopy(parent_10min)
            else:
                return deepcopy(parent)
        else:
            return deepcopy(parent)

    return deepcopy(parent)


def mutation_dsge(layer, grammar):
    """
        DSGE mutations (check DSGE for futher details)


        Parameters
        ----------
        layer : dict
            layer to be mutated (DSGE genotype)

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the
            genotype to phenotype mapping
    """

    nt_keys = sorted(list(layer.keys()))
    nt_key = random.choice(nt_keys)
    nt_idx = random.randint(0, len(layer[nt_key])-1)

    sge_possibilities = []
    random_possibilities = []
    if len(grammar.grammar[nt_key]) > 1:
        sge_possibilities = list(
            set(range(len(grammar.grammar[nt_key])))
            - set([layer[nt_key][nt_idx]['ge']])
        )
        random_possibilities.append('ge')

    if layer[nt_key][nt_idx]['ga']:
        random_possibilities.extend(['ga', 'ga'])

    if random_possibilities:
        mt_type = random.choice(random_possibilities)

        if mt_type == 'ga':
            var_name = random.choice(sorted(list(layer[nt_key][nt_idx]['ga'].keys())))
            var_type, min_val, max_val, values = layer[nt_key][nt_idx]['ga'][var_name]
            value_idx = random.randint(0, len(values)-1)

            if var_type == 'int':
                new_val = random.randint(min_val, max_val)
            elif var_type == 'float':
                new_val = values[value_idx]+random.gauss(0, 0.15)
                new_val = np.clip(new_val, min_val, max_val)

            layer[nt_key][nt_idx]['ga'][var_name][-1][value_idx] = new_val

        elif mt_type == 'ge':
            layer[nt_key][nt_idx]['ge'] = random.choice(sge_possibilities)

        else:
            return NotImplementedError


def mutation(individual, grammar, add_layer, re_use_layer, remove_layer,
             add_connection, remove_connection, dsge_layer, macro_layer,
             train_longer, default_train_time):
    """
        Network mutations: add and remove layer, add and remove connections,
        macro structure


        Parameters
        ----------
        individual : Individual
            individual to be mutated

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the
            genotype to phenotype mapping

        add_layer : float
            add layer mutation rate

        re_use_layer : float
            when adding a new layer, defines the mutation rate of using an
            already existing layer, i.e., copy by reference

        remove_layer : float
            remove layer mutation rate

        add_connection : float
            add connection mutation rate

        remove_connection : float
            remove connection mutation rate

        dsge_layer : float
            inner lever genotype mutation rate

        macro_layer : float
            inner level of the macro layers (i.e., learning, data-augmentation)
            mutation rate

        train_longer : float
            increase the training time mutation rate

        default_train_time : int
            default training time

        Returns
        -------
        ind : Individual
            mutated individual
    """

    # copy so that elite is preserved
    ind = deepcopy(individual)

    # Train individual for longer - no other mutation is applied
    if random.random() <= train_longer:
        ind.train_time += default_train_time
        return ind

    # in case the individual is mutated in any of the structural parameters
    # the training time is reseted
    ind.current_time = 0
    ind.num_epochs = 0
    ind.train_time = default_train_time

    for module in ind.modules:

        # add-layer (duplicate or new)
        for _ in range(random.randint(1, 2)):
            if len(module.layers) < module.max_expansions and \
               random.random() <= add_layer:
                if random.random() <= re_use_layer:
                    new_layer = random.choice(module.layers)
                else:
                    new_layer = grammar.initialise(module.module)

                insert_pos = random.randint(0, len(module.layers))

                # fix connections
                for _key_ in sorted(module.connections, reverse=True):
                    if _key_ >= insert_pos:
                        for value_idx, value in enumerate(module.connections[_key_]):
                            if value >= insert_pos-1:
                                module.connections[_key_][value_idx] += 1

                        module.connections[_key_+1] = module.connections.pop(_key_)

                module.layers.insert(insert_pos, new_layer)

                # make connections of the new layer
                if insert_pos == 0:
                    module.connections[insert_pos] = [-1]
                else:
                    connection_possibilities = list(
                        range(
                            max(0, insert_pos-module.levels_back),
                            insert_pos-1,
                        )
                    )
                    if len(connection_possibilities) < module.levels_back-1:
                        connection_possibilities.append(-1)

                    sample_size = random.randint(
                        0, len(connection_possibilities))

                    module.connections[insert_pos] = [insert_pos-1]
                    if sample_size > 0:
                        module.connections[insert_pos] += random.sample(
                            connection_possibilities, sample_size)

        # remove-layer
        for _ in range(random.randint(1, 2)):
            if len(module.layers) > module.min_expansions and \
               random.random() <= remove_layer:
                remove_idx = random.randint(0, len(module.layers)-1)
                del module.layers[remove_idx]

                # fix connections
                for _key_ in sorted(module.connections):
                    if _key_ > remove_idx:
                        if _key_ > remove_idx+1 and \
                           remove_idx in module.connections[_key_]:
                            module.connections[_key_].remove(remove_idx)

                        for value_idx, value in enumerate(module.connections[_key_]):
                            if value >= remove_idx:
                                module.connections[_key_][value_idx] -= 1
                        module.connections[_key_-1] = list(
                            set(module.connections.pop(_key_)))

                if remove_idx == 0:
                    module.connections[0] = [-1]

        for layer_idx, layer in enumerate(module.layers):
            # dsge mutation
            if random.random() <= dsge_layer:
                mutation_dsge(layer, grammar)

            # add connection
            if layer_idx != 0 and random.random() <= add_connection:
                connection_possibilities = list(
                    range(max(0, layer_idx-module.levels_back), layer_idx-1))
                connection_possibilities = list(
                    set(connection_possibilities)
                    - set(module.connections[layer_idx])
                )
                if len(connection_possibilities) > 0:
                    module.connections[layer_idx].append(
                        random.choice(connection_possibilities))

            # remove connection
            r_value = random.random()
            if layer_idx != 0 and r_value <= remove_connection:
                connection_possibilities = list(
                    set(module.connections[layer_idx])
                    - set([layer_idx-1])
                )
                if len(connection_possibilities) > 0:
                    r_connection = random.choice(connection_possibilities)
                    module.connections[layer_idx].remove(r_connection)

    # macro level mutation
    for macro_idx, macro in enumerate(ind.macro):
        if random.random() <= macro_layer:
            mutation_dsge(macro, grammar)

    return ind


def load_config(config_file):
    """
        Load yml configuration file.


        Parameters
        ----------
        config_file : str
            path to the configuration file

        Returns
        -------
        config : dict
            configuration dictionary
    """

    with open(Path(config_file), 'r') as f:
        config = yaml.safe_load(f)

    if config['evolutionary']['fitness_metric'] == 'accuracy':
        config['evolutionary']['fitness_function'] = fitness_metrics.accuracy
    elif config['evolutionary']['fitness_metric'] == 'mse':
        config['evolutionary']['fitness_function'] = fitness_metrics.mse
    else:
        raise ValueError(
            'Invalid fitness metric in config file: '
            f'{config["evolutionary"]["fitness_metric"]}'
        )

    return config


def main(run, dataset, config_file, grammar_path):
    """
        (1+lambda)-ES


        Parameters
        ----------
        run : int
            evolutionary run to perform

        dataset : str
            dataset to be solved

        config_file : str
            path to the configuration file

        grammar_path : str
            path to the grammar file
    """

    # load config file
    config = load_config(config_file)

    run_path = Path(config["setup"]["save_path"], f'run_{run:02d}')

    # load grammar
    grammar = Grammar(grammar_path)

    # best fitness so far
    best_fitness = None

    # load previous population content (if any)
    unpickle = unpickle_population(run_path)

    # if there is not a previous population
    if unpickle is None:
        # create directories
        makedirs(run_path, exist_ok=True)

        # set random seeds
        random.seed(config["setup"]["random_seeds"][run])
        np.random.seed(config["setup"]["numpy_seeds"][run])

        # create evaluator
        cnn_eval = Evaluator(
            dataset, config["evolutionary"]["fitness_function"])

        # save evaluator
        pickle_evaluator(cnn_eval, run_path)

        # status variables
        last_gen = -1
        total_epochs = 0

    # in case there is a previous population, load it
    else:
        last_gen, cnn_eval, population, parent, population_fits, pkl_random, \
            pkl_numpy, total_epochs = unpickle
        random.setstate(pkl_random)
        np.random.set_state(pkl_numpy)

    for gen in range(last_gen+1, config["evolutionary"]["num_generations"]):

        # check the total number of epochs (stop criteria)
        if total_epochs is not None and \
           total_epochs >= config["evolutionary"]["max_epochs"]:
            break

        if gen == 0:
            print('[%d] Creating the initial population' % (run))
            print('[%d] Performing generation: %d' % (run, gen))

            # create initial population
            population = [
                Individual(
                    config["network"]["network_structure"],
                    config["network"]["macro_structure"],
                    config["network"]["output"],
                    _id_,
                ).initialise(
                    grammar,
                    config["network"]["levels_back"],
                    config["evolutionary"]["mutations"]["reuse_layer"],
                    config["network"]["network_structure_init"],
                )
                for _id_ in range(config["evolutionary"]["lambda"])
            ]

            # set initial population variables and evaluate population
            population_fits = []
            for idx, ind in enumerate(population):
                ind.current_time = 0
                ind.num_epochs = 0
                ind.train_time = config["evolutionary"]["default_train_time"]
                population_fits.append(
                    ind.evaluate(
                        grammar,
                        cnn_eval,
                        f'{run_path}/best_{gen}_{idx}.hdf5',
                    )
                )
                ind.id = idx

        else:
            print('[%d] Performing generation: %d' % (run, gen))

            # generate offspring (by mutation)
            offspring = [
                mutation(
                    parent,
                    grammar,
                    config["evolutionary"]["mutations"]["add_layer"],
                    config["evolutionary"]["mutations"]["reuse_layer"],
                    config["evolutionary"]["mutations"]["remove_layer"],
                    config["evolutionary"]["mutations"]["add_connection"],
                    config["evolutionary"]["mutations"]["remove_connection"],
                    config["evolutionary"]["mutations"]["dsge_layer"],
                    config["evolutionary"]["mutations"]["macro_layer"],
                    config["evolutionary"]["mutations"]["train_longer"],
                    config["evolutionary"]["default_train_time"],
                )
                for _ in range(config["evolutionary"]["lambda"])
            ]

            population = [parent] + offspring

            # set elite variables to re-evaluation
            population[0].current_time = 0
            population[0].num_epochs = 0
            parent_id = parent.id

            # evaluate population
            population_fits = []
            for idx, ind in enumerate(population):
                population_fits.append(
                    ind.evaluate(
                        grammar,
                        cnn_eval,
                        f'{run_path}/best_{gen}_{idx}.hdf5',
                        f'{run_path}/best_{gen-1}_{parent_id}.hdf5',
                    )
                )
                ind.id = idx

        # select parent
        parent = select_fittest(
            population, population_fits, grammar, cnn_eval, gen, run_path,
            config["evolutionary"]["default_train_time"])

        # remove temporary files to free disk space
        if gen > 1:
            for x in range(len(population)):
                if os.path.isfile(Path(run_path, f'best_{gen-2}_{x}.hdf5')):
                    os.remove(Path(run_path, f'best_{gen-2}_{x}.hdf5'))

        # update best individual
        if best_fitness is None or parent.fitness > best_fitness:
            best_fitness = parent.fitness

            if os.path.isfile(Path(run_path, f'best_{gen}_{parent.id}.hdf5')):
                copyfile(
                    Path(run_path, f'best_{gen}_{parent.id}.hdf5'),
                    Path(run_path, 'best.hdf5'),
                )

            with open(Path(run_path, 'best_parent.pkl'), 'wb') as handle:
                pickle.dump(parent, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'[{run}] Best fitness of generation {gen}: '
              f'{max(population_fits)}')
        print(f'[{run}] Best overall fitness: {best_fitness}')

        # save population
        save_pop(population, run_path, gen)
        pickle_population(population, parent, run_path)

        total_epochs += sum([ind.num_epochs for ind in population])

    # compute testing performance of the fittest network
    best_test_acc = cnn_eval.testing_performance(
        str(Path(run_path, 'best.hdf5')))
    print('[%d] Best test accuracy: %f' % (run, best_test_acc))


def process_input(argv):
    """
        Maps and checks the input parameters and call the main function.

        Parameters
        ----------
        argv : list
            argv from system
    """

    dataset = None
    config_file = None
    run = 0
    grammar = None

    try:
        opts, args = getopt.getopt(
            argv,
            "hd:c:r:g:",
            ["dataset=", "config=", "run=", "grammar="],
        )
    except getopt.GetoptError:
        print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammra>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammra>')
            sys.exit()

        elif opt in ("-d", "--dataset"):
            dataset = arg

        elif opt in ("-c", "--config"):
            config_file = arg

        elif opt in ("-r", "--run"):
            run = int(arg)

        elif opt in ("-g", "--grammar"):
            grammar = arg

    error = False

    # check if mandatory variables are all set
    if dataset is None:
        print('The dataset (-d) parameter is mandatory.')
        error = True

    if config_file is None:
        print('The config. file parameter (-c) is mandatory.')
        error = True

    if grammar is None:
        print('The grammar (-g) parameter is mandatory.')
        error = True

    if error:
        print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')
        exit(-1)

    # check if files exist
    if not os.path.isfile(grammar):
        print('Grammar file does not exist.')
        error = True

    if not os.path.isfile(config_file):
        print('Configuration file does not exist.')
        error = True

    if not error:
        main(run, dataset, config_file, grammar)
    else:
        print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')


if __name__ == '__main__':
    import getopt
    import sys

    process_input(sys.argv[1:])
