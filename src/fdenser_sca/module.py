import random


class Module:
    """
        Each of the units of the outer-level genotype


        Attributes
        ----------
        module : str
            non-terminal symbol

        min_expansions : int
            minimum expansions of the block

        max_expansions : int
            maximum expansions of the block

        levels_back : dict
            number of previous layers a given layer can receive as input

        layers : list
            list of layers of the module

        connections : dict
            list of connections of each layer


        Methods
        -------
            initialise(grammar, reuse)
                Randomly creates a module
    """

    def __init__(self, module, min_expansions, max_expansions,
                 levels_back, min_expansins):
        """
            Parameters
            ----------
            module : str
                non-terminal symbol

            min_expansions : int
                minimum expansions of the block

            max_expansions : int
                maximum expansions of the block

            levels_back : dict
                number of previous layers a given layer can receive as input
        """

        self.module = module
        self.min_expansions = min_expansins
        self.max_expansions = max_expansions
        self.levels_back = levels_back
        self.layers = []
        self.connections = {}

    def initialise(self, grammar, reuse, init_max):
        """
            Randomly creates a module

            Parameters
            ----------
            grammar : Grammar
                grammar instace that stores the expansion rules

            reuse : float
                likelihood of reusing an existing layer

            Returns
            -------
            score_history : dict
                training data: loss and accuracy
        """

        num_expansions = random.choice(init_max[self.module])

        # Initialise layers
        for idx in range(num_expansions):
            if idx > 0 and random.random() <= reuse:
                r_idx = random.randint(0, idx-1)
                self.layers.append(self.layers[r_idx])
            else:
                self.layers.append(grammar.initialise(self.module))

        # Initialise connections: feed-forward and allowing skip-connections
        self.connections = {}
        for layer_idx in range(num_expansions):
            if layer_idx == 0:
                # the -1 layer is the input
                self.connections[layer_idx] = [-1,]
            else:
                connection_possibilities = list(
                    range(max(0, layer_idx-self.levels_back), layer_idx-1)
                )
                if len(connection_possibilities) < self.levels_back-1:
                    connection_possibilities.append(-1)

                sample_size = random.randint(0, len(connection_possibilities))

                self.connections[layer_idx] = [layer_idx-1]
                if sample_size > 0:
                    self.connections[layer_idx] += random.sample(
                        connection_possibilities, sample_size
                    )
