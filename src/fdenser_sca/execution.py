import json
import os
import pickle
import random
from glob import glob
from pathlib import Path

import numpy as np


def save_pop(population, run_path, gen):
    """
        Save the current population statistics in json.
        For each individual:
            .id: unique generation identifier
            .phenotype: phenotype of the individual
            .fitness: fitness of the individual
            .metrics: other evaluation metrics (e.g., loss, accuracy)
            .trainable_parameters: number of network trainable parameters
            .num_epochs: number of performed training epochs
            .time: time (sec) the network took to perform num_epochs
            .train_time: maximum time (sec) that the network is allowed to
                         train for



        Parameters
        ----------
        population : list
            list of Individual instances

        save_path : str
            path to the json file

        run : int
            current evolutionary run

        gen : int
            current generation

    """

    json_dump = []

    for ind in population:
        json_dump.append({
                          'id': ind.id,
                          'phenotype': ind.phenotype,
                          'fitness': ind.fitness,
                          'metrics': ind.metrics,
                          'trainable_parameters': ind.trainable_parameters,
                          'num_epochs': ind.num_epochs,
                          'time': ind.time,
                          'train_time': ind.train_time})

    with open(Path(f'{run_path}/gen_{gen:02d}.json'), 'w') as f_json:
        f_json.write(json.dumps(json_dump, indent=4))


def pickle_evaluator(evaluator, run_path):
    """
        Save the Evaluator instance to later enable resuming evolution

        Parameters
        ----------
        evaluator : Evaluator
            instance of the Evaluator class

        save_path: str
            path to the json file

        run : int
            current evolutionary run

    """

    with open(Path(f'{run_path}/evaluator.pkl'), 'wb') as handle:
        pickle.dump(evaluator, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_population(population, parent, run_path):
    """
        Save the objects (pickle) necessary to later resume evolution:
        Pickled objects:
            .population
            .parent
            .random states: numpy and random
        Useful for later conducting more generations.
        Replaces the objects of the previous generation.

        Parameters
        ----------
        population : list
            list of Individual instances

        parent : Individual
            fittest individual that will seed the next generation

        save_path: str
            path to the json file

        run : int
            current evolutionary run
    """

    with open(Path(f'{run_path}/population.pkl'), 'wb') as handle_pop:
        pickle.dump(population, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path(f'{run_path}/parent.pkl'), 'wb') as handle_pop:
        pickle.dump(parent, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path(f'{run_path}/random.pkl'), 'wb') as handle_random:
        pickle.dump(
            random.getstate(), handle_random, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path(f'{run_path}/numpy.pkl'), 'wb') as handle_numpy:
        pickle.dump(
            np.random.get_state(),
            handle_numpy,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def get_total_epochs(run_path, last_gen):
    """
        Compute the total number of performed epochs.

        Parameters
        ----------
        save_path: str
            path where the ojects needed to resume evolution are stored.

        run : int
            current evolutionary run

        last_gen : int
            count the number of performed epochs until the last_gen generation


        Returns
        -------
        total_epochs : int
            sum of the number of epochs performed by all trainings
    """

    total_epochs = 0
    for gen in range(0, last_gen+1):
        j = json.load(open(Path(f'{run_path}/gen_{gen:02d}.json')))
        num_epochs = [elm['num_epochs'] for elm in j]
        total_epochs += sum(num_epochs)

    return total_epochs


def unpickle_population(run_path):
    """
        Save the objects (pickle) necessary to later resume evolution.
        Useful for later conducting more generations.
        Replaces the objects of the previous generation.
        Returns None in case any generation has been performed yet.


        Parameters
        ----------
        save_path: str
            path where the ojects needed to resume evolution are stored.

        run : int
            current evolutionary run


        Returns
        -------
        last_generation : int
            idx of the last performed generation

        pickle_evaluator : Evaluator
            instance of the Evaluator class used for evaluating the
            individuals. Loaded because it has the data used for training.

        pickle_population : list
            population of the last performed generation

        pickle_parent : Individual
            fittest individual of the last performed generation

        pickle_population_fitness : list
            ordered list of fitnesses of the last population of individuals

        pickle_random : tuple
            Random random state

        pickle_numpy : tuple
            Numpy random state
    """

    csvs = glob(str(Path(run_path, '*.json')))

    if csvs:
        csvs = [
            int(csv.split(os.sep)[-1].replace('gen_', '').replace('.json', ''))
            for csv in csvs
        ]
        last_generation = max(csvs)

        with open(Path(f'{run_path}/evaluator.pkl'), 'rb') as handle_eval:
            pickle_evaluator = pickle.load(handle_eval)

        with open(Path(f'{run_path}/population.pkl'), 'rb') as handle_pop:
            pickle_population = pickle.load(handle_pop)

        with open(Path(f'{run_path}/parent.pkl'), 'rb') as handle_parent:
            pickle_parent = pickle.load(handle_parent)

        pickle_population_fitness = [ind.fitness for ind in pickle_population]

        with open(Path(f'{run_path}/random.pkl'), 'rb') as handle_random:
            pickle_random = pickle.load(handle_random)

        with open(Path(f'{run_path}/numpy.pkl'), 'rb') as handle_numpy:
            pickle_numpy = pickle.load(handle_numpy)

        total_epochs = get_total_epochs(run_path, last_generation)

        return last_generation, pickle_evaluator, pickle_population, \
            pickle_parent, pickle_population_fitness, pickle_random, \
            pickle_numpy, total_epochs

    else:
        return None
