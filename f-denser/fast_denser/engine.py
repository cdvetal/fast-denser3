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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sys import argv
import random
import numpy as np
from fast_denser.grammar import Grammar
from fast_denser.utils import Evaluator, Individual
from copy import deepcopy
from os import makedirs
import pickle
import os
from shutil import copyfile
from glob import glob
import json
from keras.preprocessing.image import ImageDataGenerator
from fast_denser.utilities.fitness_metrics import * 
from jsmin import jsmin
from fast_denser.utilities.data_augmentation import augmentation
from pathlib import Path

def save_pop(population, save_path, run, gen):
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
            .train_time: maximum time (sec) that the network is allowed to train for



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

    with open(Path('%s/run_%d/gen_%d.csv' % (save_path, run, gen)), 'w') as f_json:
        f_json.write(json.dumps(json_dump, indent=4))



def pickle_evaluator(evaluator, save_path, run):
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

    with open(Path('%s/run_%d/evaluator.pkl' % (save_path, run)), 'wb') as handle:
        pickle.dump(evaluator, handle, protocol=pickle.HIGHEST_PROTOCOL)



def pickle_population(population, parent, save_path, run):
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

    with open(Path('%s/run_%d/population.pkl' % (save_path, run)), 'wb') as handle_pop:
        pickle.dump(population, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path('%s/run_%d/parent.pkl' % (save_path, run)), 'wb') as handle_pop:
        pickle.dump(parent, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path('%s/run_%d/random.pkl' % (save_path, run)), 'wb') as handle_random:
        pickle.dump(random.getstate(), handle_random, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path('%s/run_%d/numpy.pkl' % (save_path, run)), 'wb') as handle_numpy:
        pickle.dump(np.random.get_state(), handle_numpy, protocol=pickle.HIGHEST_PROTOCOL)



def get_total_epochs(save_path, run, last_gen):
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
        j = json.load(open(Path('%s/run_%d/gen_%d.csv' % (save_path, run, gen))))
        num_epochs = [elm['num_epochs'] for elm in j]
        total_epochs += sum(num_epochs)

    return total_epochs



def unpickle_population(save_path, run):
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
            instance of the Evaluator class used for evaluating the individuals.
            Loaded because it has the data used for training.

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

    csvs = glob(str(Path('%s' % save_path, 'run_%d' % run, '*.csv' )))
    
    if csvs:
        csvs = [int(csv.split(os.sep)[-1].replace('gen_','').replace('.csv','')) for csv in csvs]
        last_generation = max(csvs)

        with open(Path('%s' % save_path, 'run_%d' % run, 'evaluator.pkl'), 'rb') as handle_eval:
            pickle_evaluator = pickle.load(handle_eval)

        with open(Path('%s' % save_path, 'run_%d' % run, 'population.pkl'), 'rb') as handle_pop:
            pickle_population = pickle.load(handle_pop)

        with open(Path('%s' % save_path, 'run_%d' % run, 'parent.pkl'), 'rb') as handle_parent:
            pickle_parent = pickle.load(handle_parent)

        pickle_population_fitness = [ind.fitness for ind in pickle_population]

        with open(Path('%s' % save_path, 'run_%d' % run, 'random.pkl'), 'rb') as handle_random:
            pickle_random = pickle.load(handle_random)

        with open(Path('%s' % save_path, 'run_%d' % run, 'numpy.pkl'), 'rb') as handle_numpy:
            pickle_numpy = pickle.load(handle_numpy)

        total_epochs = get_total_epochs(save_path, run, last_generation)

        return last_generation, pickle_evaluator, pickle_population, pickle_parent, \
               pickle_population_fitness, pickle_random, pickle_numpy, total_epochs

    else:
        return None



def select_fittest(population, population_fits, grammar, cnn_eval, datagen, datagen_test, gen, save_path, default_train_time): #pragma: no cover
    """
        Select the parent to seed the next generation.


        Parameters
        ----------
        population : list
            list of instances of Individual

        population_fits : list
            ordered list of fitnesses of the population of individuals

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the genotype
            to phenotype mapping

        cnn_eval : Evaluator
            Evaluator instance used to train the networks

        datagen : keras.preprocessing.image.ImageDataGenerator
            Data augmentation method image data generator for the training data

        datagen_test : keras.preprocessing.image.ImageDataGenerator
            Data augmentation method image data generator for the validation and test data

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


    #Get best individual just according to fitness
    idx_max = np.argmax(population_fits)
    parent = population[idx_max]    

    #however if the parent is not the elite, and the parent is trained for longer, the elite
    #is granted the same evaluation time.
    if parent.train_time > default_train_time:
        retrain_elite = False
        if idx_max != 0 and population[0].train_time > default_train_time and population[0].train_time < parent.train_time:
            retrain_elite = True
            elite = population[0]
            elite.train_time = parent.train_time
            elite.evaluate(grammar, cnn_eval, datagen, datagen_test, '%s/best_%d_%d.hdf5' % (save_path, gen, elite.id), '%s/best_%d_%d.hdf5' % (save_path, gen, elite.id))
            population_fits[0] = elite.fitness

        min_train_time = min([ind.current_time for ind in population])

        #also retrain the best individual that is trained just for the default time
        retrain_10min = False
        if min_train_time < parent.train_time:
            ids_10min = [ind.current_time == min_train_time for ind in population]
    
            if sum(ids_10min) > 0:
                retrain_10min = True
                indvs_10min = np.array(population)[ids_10min]
                max_fitness_10min = max([ind.fitness for ind in indvs_10min])
                idx_max_10min = np.argmax(max_fitness_10min)
                parent_10min = indvs_10min[idx_max_10min]

                parent_10min.train_time = parent.train_time

                parent_10min.evaluate(grammar, cnn_eval, datagen, datagen_test, '%s/best_%d_%d.hdf5' % (save_path, gen, parent_10min.id), '%s/best_%d_%d.hdf5' % (save_path, gen, parent_10min.id))

                population_fits[population.index(parent_10min)] = parent_10min.fitness


        #select the fittest amont all retrains and the initial parent
        if retrain_elite:
            if retrain_10min:
                if parent_10min.fitness > elite.fitness and parent_10min.fitness > parent.fitness:
                    return deepcopy(parent_10min)
                elif elite.fitness > parent_10min.fitness and elite.fitness > parent.fitness:
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
            Grammar instance, used to perform the initialisation and the genotype
            to phenotype mapping
    """

    nt_keys = sorted(list(layer.keys()))
    nt_key = random.choice(nt_keys)
    nt_idx = random.randint(0, len(layer[nt_key])-1)

    sge_possibilities = []
    random_possibilities = []
    if len(grammar.grammar[nt_key]) > 1:
        sge_possibilities = list(set(range(len(grammar.grammar[nt_key]))) -\
                                 set([layer[nt_key][nt_idx]['ge']]))
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



def mutation(individual, grammar, add_layer, re_use_layer, remove_layer, add_connection,\
             remove_connection, dsge_layer, macro_layer, train_longer, default_train_time):
    """
        Network mutations: add and remove layer, add and remove connections, macro structure


        Parameters
        ----------
        individual : Individual
            individual to be mutated

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the genotype
            to phenotype mapping

        add_layer : float
            add layer mutation rate

        re_use_layer : float
            when adding a new layer, defines the mutation rate of using an already
            existing layer, i.e., copy by reference

        remove_layer : float
            remove layer mutation rate

        add_connection : float
            add connection mutation rate

        remove_connection : float
            remove connection mutation rate

        dsge_layer : float
            inner lever genotype mutation rate

        macro_layer : float
            inner level of the macro layers (i.e., learning, data-augmentation) mutation rate

        train_longer : float
            increase the training time mutation rate

        default_train_time : int
            default training time

        Returns
        -------
        ind : Individual
            mutated individual
    """

    #copy so that elite is preserved
    ind = deepcopy(individual)

    #Train individual for longer - no other mutation is applied
    if random.random() <= train_longer:
        ind.train_time += default_train_time
        return ind


    #in case the individual is mutated in any of the structural parameters
    #the training time is reseted
    ind.current_time = 0
    ind.num_epochs = 0
    ind.train_time = default_train_time
    
    for module in ind.modules:

        #add-layer (duplicate or new)
        for _ in range(random.randint(1,2)):
            if len(module.layers) < module.max_expansions and random.random() <= add_layer:
                if random.random() <= re_use_layer:
                    new_layer = random.choice(module.layers)
                else:
                    new_layer = grammar.initialise(module.module)

                insert_pos = random.randint(0, len(module.layers))

                #fix connections
                for _key_ in sorted(module.connections, reverse=True):
                    if _key_ >= insert_pos:
                        for value_idx, value in enumerate(module.connections[_key_]):
                            if value >= insert_pos-1:
                                module.connections[_key_][value_idx] += 1

                        module.connections[_key_+1] = module.connections.pop(_key_)


                module.layers.insert(insert_pos, new_layer)

                #make connections of the new layer
                if insert_pos == 0:
                    module.connections[insert_pos] = [-1]
                else:
                    connection_possibilities = list(range(max(0, insert_pos-module.levels_back), insert_pos-1))
                    if len(connection_possibilities) < module.levels_back-1:
                        connection_possibilities.append(-1)

                    sample_size = random.randint(0, len(connection_possibilities))
                    
                    module.connections[insert_pos] = [insert_pos-1] 
                    if sample_size > 0:
                        module.connections[insert_pos] += random.sample(connection_possibilities, sample_size)


        #remove-layer
        for _ in range(random.randint(1,2)):
            if len(module.layers) > module.min_expansions and random.random() <= remove_layer:
                remove_idx = random.randint(0, len(module.layers)-1)
                del module.layers[remove_idx]
                
                #fix connections
                for _key_ in sorted(module.connections):
                    if _key_ > remove_idx:
                        if _key_ > remove_idx+1 and remove_idx in module.connections[_key_]:
                            module.connections[_key_].remove(remove_idx)

                        for value_idx, value in enumerate(module.connections[_key_]):
                            if value >= remove_idx:
                                module.connections[_key_][value_idx] -= 1
                        module.connections[_key_-1] = list(set(module.connections.pop(_key_)))

                if remove_idx == 0:
                    module.connections[0] = [-1]


        for layer_idx, layer in enumerate(module.layers):
            #dsge mutation
            if random.random() <= dsge_layer:
                mutation_dsge(layer, grammar)

            #add connection
            if layer_idx != 0 and random.random() <= add_connection:
                connection_possibilities = list(range(max(0, layer_idx-module.levels_back), layer_idx-1))
                connection_possibilities = list(set(connection_possibilities) - set(module.connections[layer_idx]))
                if len(connection_possibilities) > 0:
                    module.connections[layer_idx].append(random.choice(connection_possibilities))

            #remove connection
            r_value = random.random()
            if layer_idx != 0 and r_value <= remove_connection:
                connection_possibilities = list(set(module.connections[layer_idx]) - set([layer_idx-1]))
                if len(connection_possibilities) > 0:
                    r_connection = random.choice(connection_possibilities)
                    module.connections[layer_idx].remove(r_connection)


    #macro level mutation
    for macro_idx, macro in enumerate(ind.macro): 
        if random.random() <= macro_layer:
            mutation_dsge(macro, grammar)
                    

    return ind



def load_config(config_file):
    """
        Load configuration json file.


        Parameters
        ----------
        config_file : str
            path to the configuration file
            
        Returns
        -------
        config : dict
            configuration json file
    """

    with open(Path(config_file)) as js_file:
        minified = jsmin(js_file.read())

    config = json.loads(minified)

    config["TRAINING"]["datagen"] = eval(config["TRAINING"]["datagen"])
    config["TRAINING"]["datagen_test"] = eval(config["TRAINING"]["datagen_test"])
    config["TRAINING"]["fitness_metric"] = eval(config["TRAINING"]["fitness_metric"])

    return config



def main(run, dataset, config_file, grammar_path): #pragma: no cover
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

    #load config file
    config = load_config(config_file)

    #load grammar
    grammar = Grammar(grammar_path)

    #best fitness so far
    best_fitness = None

    #load previous population content (if any)
    unpickle = unpickle_population(config["EVOLUTIONARY"]["save_path"], run)

    #if there is not a previous population
    if unpickle is None:
        #create directories
        makedirs('%s/run_%d/' % (config["EVOLUTIONARY"]["save_path"], run), exist_ok=True)

        #set random seeds
        random.seed(config["EVOLUTIONARY"]["random_seeds"][run])
        np.random.seed(config["EVOLUTIONARY"]["numpy_seeds"][run])

        #create evaluator
        cnn_eval = Evaluator(dataset, config["TRAINING"]["fitness_metric"])

        #save evaluator
        pickle_evaluator(cnn_eval, config["EVOLUTIONARY"]["save_path"], run)

        #status variables
        last_gen = -1
        total_epochs = 0
    
    #in case there is a previous population, load it
    else:
        last_gen, cnn_eval, population, parent, population_fits, pkl_random, pkl_numpy, total_epochs = unpickle
        random.setstate(pkl_random)
        np.random.set_state(pkl_numpy)


    for gen in range(last_gen+1, config["EVOLUTIONARY"]["num_generations"]):

        #check the total number of epochs (stop criteria)
        if total_epochs is not None and total_epochs >= config["EVOLUTIONARY"]["max_epochs"]:
            break

        if gen == 0:
            print('[%d] Creating the initial population' % (run))
            print('[%d] Performing generation: %d' % (run, gen))
            
            #create initial population
            population = [Individual(config["NETWORK"]["network_structure"], config["NETWORK"]["macro_structure"],\
                          config["NETWORK"]["output"], _id_).initialise(grammar, config["NETWORK"]["levels_back"],\
                          config["EVOLUTIONARY"]["MUTATIONS"]["reuse_layer"], config["NETWORK"]["network_structure_init"]) \
                          for _id_ in range(config["EVOLUTIONARY"]["lambda"])]

            #set initial population variables and evaluate population
            population_fits = []
            for idx, ind in enumerate(population):
                ind.current_time = 0
                ind.num_epochs = 0
                ind.train_time = config["TRAINING"]["default_train_time"]
                population_fits.append(ind.evaluate(grammar, cnn_eval, config["TRAINING"]["datagen"], config["TRAINING"]["datagen_test"], '%s/run_%d/best_%d_%d.hdf5' % (config["EVOLUTIONARY"]["save_path"], run, gen, idx)))
                ind.id = idx
        
        else:
            print('[%d] Performing generation: %d' % (run, gen))
            
            #generate offspring (by mutation)
            offspring = [mutation(parent, grammar, config["EVOLUTIONARY"]["MUTATIONS"]["add_layer"],
                                  config["EVOLUTIONARY"]["MUTATIONS"]["reuse_layer"], config["EVOLUTIONARY"]["MUTATIONS"]["remove_layer"], 
                                  config["EVOLUTIONARY"]["MUTATIONS"]["add_connection"], config["EVOLUTIONARY"]["MUTATIONS"]["remove_connection"],
                                  config["EVOLUTIONARY"]["MUTATIONS"]["dsge_layer"], config["EVOLUTIONARY"]["MUTATIONS"]["macro_layer"],
                                  config["EVOLUTIONARY"]["MUTATIONS"]["train_longer"], config["TRAINING"]["default_train_time"]) 
                                  for _ in range(config["EVOLUTIONARY"]["lambda"])]

            population = [parent] + offspring

            #set elite variables to re-evaluation
            population[0].current_time = 0
            population[0].num_epochs = 0
            parent_id = parent.id

            #evaluate population
            population_fits = []
            for idx, ind in enumerate(population):
                population_fits.append(ind.evaluate(grammar, cnn_eval, config["TRAINING"]["datagen"], config["TRAINING"]["datagen_test"], '%s/run_%d/best_%d_%d.hdf5' % (config["EVOLUTIONARY"]["save_path"], run, gen, idx), '%s/run_%d/best_%d_%d.hdf5' % (config["EVOLUTIONARY"]["save_path"], run, gen-1, parent_id)))
                ind.id = idx

        #select parent
        parent = select_fittest(population, population_fits, grammar, cnn_eval,\
                                config["TRAINING"]["datagen"], config["TRAINING"]["datagen_test"], gen, \
                                config["EVOLUTIONARY"]["save_path"]+'/run_'+str(run),\
                                config["TRAINING"]["default_train_time"])

        #remove temporary files to free disk space
        if gen > 1:
            for x in range(len(population)):
                if os.path.isfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.hdf5' % (gen-2, x))):
                    os.remove(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.hdf5' % (gen-2, x)))
                    os.remove(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.h5' % (gen-2, x)))

        #update best individual
        if best_fitness is None or parent.fitness > best_fitness:
            best_fitness = parent.fitness

            if os.path.isfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.hdf5' % (gen, parent.id))):
                copyfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.hdf5' % (gen, parent.id)), Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best.hdf5'))
                copyfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.h5' % (gen, parent.id)), Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best.h5'))
            
            with open('%s/run_%d/best_parent.pkl' % (config["EVOLUTIONARY"]["save_path"], run), 'wb') as handle:
                pickle.dump(parent, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('[%d] Best fitness of generation %d: %f' % (run, gen, max(population_fits)))
        print('[%d] Best overall fitness: %f' % (run, best_fitness))

        #save population
        save_pop(population, config["EVOLUTIONARY"]["save_path"], run, gen)
        pickle_population(population, parent, config["EVOLUTIONARY"]["save_path"], run)

        total_epochs += sum([ind.num_epochs for ind in population])


    #compute testing performance of the fittest network
    best_test_acc = cnn_eval.testing_performance(str(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best.h5')), config["TRAINING"]["datagen_test"])
    print('[%d] Best test accuracy: %f' % (run, best_test_acc))



def process_input(argv): #pragma: no cover
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
        opts, args = getopt.getopt(argv, "hd:c:r:g:",["dataset=","config=","run=","grammar="]   )
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

    #check if mandatory variables are all set
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

    #check if files exist
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



if __name__ == '__main__': #pragma: no cover
    import sys, getopt

    process_input(sys.argv[1:]) 
