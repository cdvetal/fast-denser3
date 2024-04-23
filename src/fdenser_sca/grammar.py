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


from random import randint, uniform


class Grammar:
    """
        Dynamic Structured Grammatical Evolution (DSGE) code. F-DENSER++ uses
        a BNF grammar to define the search space, and DSGE is applied to
        perform the genotype/phenotype mapping of the inner-level of the
        genotype.


        Attributes
        ----------
        grammar : dict
            object where the grammar is stored, and later used for
            initialisation, and decoding of the individuals.


        Methods
        -------
        get_grammar(path)
            Reads the grammar from a file

        read_grammar(path)
            Auxiliary function of the get_grammar method; loads the grammar
            from a file

        parse_grammar(path)
            Auxiliary fuction of the get_grammar method; parses the grammar
            to a dictionary

        _str_()
            Prints the grammar in the BNF form

        initialise(start_symbol)
            Creates a genotype, at random, starting from the input non-terminal
            symbol

        initialise_recursive(symbol, prev_nt, genotype)
            Auxiliary function of the initialise method; recursively expands
            the non-terminal symbol

        decode(start_symbol, genotype)
            Genotype to phenotype mapping.

        decode_recursive(symbol, read_integers, genotype, phenotype)
            Auxiliary function of the decode method; recursively applies the
            expansions that are encoded in the genotype
    """

    def __init__(self, path):
        """
            Parameters
            ----------
            path : str
                Path to the BNF grammar file
        """

        self.grammar = self.get_grammar(path)

    def get_grammar(self, path):
        """
            Read the grammar from a file.

            Parameters
            ----------
            path : str
                Path to the BNF grammar file

            Returns
            -------
            grammar : dict
                object where the grammar is stored, and later used for
                initialisation, and decoding of the individuals
        """

        raw_grammar = self.read_grammar(path)

        if raw_grammar is None:
            print('Grammar file does not exist.')
            exit(-1)

        return self.parse_grammar(raw_grammar)

    def read_grammar(self, path):
        """
            Auxiliary function of the get_grammar method; loads the grammar
            from a file

            Parameters
            ----------
            path : str
                Path to the BNF grammar file

            Returns
            -------
            raw_grammar : list
                list of strings, where each position is a line of the grammar
                file. Returns None in case of failure opening the file.
        """

        try:
            with open(path, 'r') as f_in:
                raw_grammar = f_in.readlines()
                return raw_grammar
        except IOError:
            return None

    def parse_grammar(self, raw_grammar):
        """
            Auxiliary fuction of the get_grammar method; parses the grammar
            to a dictionary

            Parameters
            ----------
            raw_grammar : list
                list of strings, where each position is a line of the grammar
                file

            Returns
            -------
            grammar : dict
                object where the grammar is stored, and later used for
                initialisation, and decoding of the individuals
        """

        grammar = {}
        start_symbol = None

        for rule in raw_grammar:
            non_terminal, raw_rule_expansions = rule.rstrip('\n').split('::=')

            rule_expansions = []
            for production_rule in raw_rule_expansions.split('|'):
                rule_expansions.append([
                    (symbol.rstrip().lstrip().replace('<', '')
                     .replace('>', ''), '<' in symbol)
                    for symbol in production_rule.rstrip().lstrip().split(' ')
                ])
            grammar[non_terminal.rstrip().lstrip().replace('<', '').replace('>', '')] = rule_expansions

            if start_symbol is None:
                start_symbol = non_terminal.rstrip().lstrip().replace('<', '').replace('>', '')

        return grammar

    def _str_(self):
        """
        Prints the grammar in the BNF form
        """

        for _key_ in sorted(self.grammar):
            productions = ''
            for production in self.grammar[_key_]:
                for symbol, terminal in production:
                    if terminal:
                        productions += ' <'+symbol+'>'
                    else:
                        productions += ' '+symbol
                productions += ' | '
            print('<'+_key_+'> ::='+productions[:-3])

    def __str__(self):
        """
        Prints the grammar in the BNF form
        """

        print_str = ''
        for _key_ in sorted(self.grammar):
            productions = ''
            for production in self.grammar[_key_]:
                for symbol, terminal in production:
                    if terminal:
                        productions += ' <'+symbol+'>'
                    else:
                        productions += ' '+symbol
                productions += ' | '
            print_str += '<'+_key_+'> ::='+productions[:-3]+'\n'

        return print_str

    def initialise(self, start_symbol):
        """
            Creates a genotype, at random, starting from the input non-terminal
            symbol

            Parameters
            ----------
            start_symbol : str
                non-terminal symbol used as starting symbol for the grammatical
                expansion.

            Returns
            -------
            genotype : dict
                DSGE genotype used for the inner-level of F-DENSER++
        """

        genotype = {}

        self.initialise_recursive((start_symbol, True), None, genotype)

        return genotype

    def initialise_recursive(self, symbol, prev_nt, genotype):
        """
            Auxiliary function of the initialise method; recursively expands
            the non-terminal symbol

            Parameters
            ----------
            symbol : tuple
                (non terminal symbol to expand : str, non-terminal : bool).
                Non-terminal is True in case the non-terminal symbol is a
                non-terminal, and False if the the non-terminal symbol str is
                a terminal

            prev_nt: str
                non-terminal symbol used in the previous expansion

            genotype: dict
                DSGE genotype used for the inner-level of F-DENSER++

        """

        symbol, non_terminal = symbol

        if non_terminal:
            expansion_possibility = randint(0, len(self.grammar[symbol])-1)

            if symbol not in genotype:
                genotype[symbol] = [{'ge': expansion_possibility, 'ga': {}}]
            else:
                genotype[symbol].append({'ge': expansion_possibility, 'ga': {}})

            add_reals_idx = len(genotype[symbol])-1
            for sym in self.grammar[symbol][expansion_possibility]:
                self.initialise_recursive(sym, (symbol, add_reals_idx), genotype)
        else:
            if '[' in symbol and ']' in symbol:
                genotype_key, genotype_idx = prev_nt

                [var_name, var_type, num_values, min_val, max_val] = \
                    symbol.replace('[', '').replace(']', '').split(',')

                num_values = int(num_values)
                min_val, max_val = float(min_val), float(max_val)

                if var_type == 'int':
                    values = [randint(min_val, max_val) for _ in range(num_values)]
                elif var_type == 'float':
                    values = [uniform(min_val, max_val) for _ in range(num_values)]

                genotype[genotype_key][genotype_idx]['ga'][var_name] = \
                    (var_type, min_val, max_val, values)

    def decode(self, start_symbol, genotype):
        """
            Genotype to phenotype mapping.

            Parameters
            ----------
            start_symbol : str
                non-terminal symbol used as starting symbol for the grammatical
                expansion

            genotype : dict
                DSGE genotype used for the inner-level of F-DENSER++

            Returns
            -------
            phenotype : str
                phenotype corresponding to the input genotype
        """

        read_codons = dict.fromkeys(list(genotype.keys()), 0)
        phenotype = self.decode_recursive(
            (start_symbol, True), read_codons, genotype, '')

        return phenotype.lstrip().rstrip()

    def decode_recursive(self, symbol, read_integers, genotype, phenotype):
        """
            Auxiliary function of the decode method; recursively applies the
            expansions that are encoded in the genotype

            Parameters
            ----------
            symbol : tuple
                (non terminal symbol to expand : str, non-terminal : bool).
                Non-terminal is True in case the non-terminal symbol is a
                non-terminal, and False if the the non-terminal symbol str is
                a terminal

            read_integers : dict
                index of the next codon of the non-terminal genotype to be read

            genotype : dict
                DSGE genotype used for the inner-level of F-DENSER++

            phenotype : str
                phenotype corresponding to the input genotype
        """

        symbol, non_terminal = symbol

        if non_terminal:
            if symbol not in read_integers:
                read_integers[symbol] = 0
                genotype[symbol] = []

            if len(genotype[symbol]) <= read_integers[symbol]:
                ge_expansion_integer = randint(0, len(self.grammar[symbol])-1)
                genotype[symbol].append({'ge': ge_expansion_integer, 'ga': {}})

            current_nt = read_integers[symbol]
            expansion_integer = genotype[symbol][current_nt]['ge']
            read_integers[symbol] += 1
            expansion = self.grammar[symbol][expansion_integer]

            used_terminals = []
            for sym in expansion:
                if sym[1]:
                    phenotype = self.decode_recursive(
                        sym, read_integers, genotype, phenotype)
                else:
                    if '[' in sym[0] and ']' in sym[0]:
                        var_name, var_type, var_num_values, var_min, var_max = \
                            sym[0].replace('[', '').replace(']', '').split(',')
                        if var_name not in genotype[symbol][current_nt]['ga']:
                            var_num_values = int(var_num_values)
                            var_min, var_max = float(var_min), float(var_max)

                            if var_type == 'int':
                                values = [randint(var_min, var_max)
                                          for _ in range(var_num_values)]
                            elif var_type == 'float':
                                values = [uniform(var_min, var_max)
                                          for _ in range(var_num_values)]

                            genotype[symbol][current_nt]['ga'][var_name] = (
                                var_type, var_min, var_max, values)

                        values = genotype[symbol][current_nt]['ga'][var_name][-1]

                        phenotype += f' {var_name}:{",".join(map(str, values))}'

                        used_terminals.append(var_name)
                    else:
                        phenotype += ' ' + sym[0]

            unused_terminals = list(
                set(list(genotype[symbol][current_nt]['ga'].keys()))
                - set(used_terminals)
                )
            if unused_terminals:
                for name in used_terminals:
                    del genotype[symbol][current_nt]['ga'][name]

        return phenotype
