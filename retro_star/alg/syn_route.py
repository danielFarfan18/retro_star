import numpy as np
from queue import Queue
from graphviz import Digraph

class SynRoute:
    """
    Represents a synthetic route for a target molecule.

    Attributes:
        target_mol (str): The target molecule for the synthetic route.
        mols (list): A list of molecules in the route.
        values (list): A list of values associated with each molecule.
        templates (list): A list of templates associated with each molecule.
        parents (list): A list of parent molecule indices for each molecule.
        children (list): A list of child molecule indices for each molecule.
        optimal (bool): Indicates if the route is optimal.
        costs (dict): A dictionary of costs associated with each molecule.
        succ_value (float): The success value of the route.
        total_cost (float): The total cost of the route.
        length (int): The length of the route.
        search_status (float): The search status of the route.
    """

    def __init__(self, target_mol, succ_value, search_status):
        """
        Initializes a SynRoute object.

        Args:
            target_mol (str): The target molecule for the synthetic route.
            succ_value (float): The success value of the route.
            search_status (float): The search status of the route.
        """
        self.target_mol = target_mol
        self.mols = [target_mol]
        self.values = [None]
        self.templates = [None]
        self.parents = [-1]
        self.children = [None]
        self.optimal = False
        self.costs = {}

        self.succ_value = succ_value
        self.total_cost = 0
        self.length = 0
        self.search_status = search_status
        if self.succ_value <= self.search_status:
            self.optimal = True

    def _add_mol(self, mol, parent_id):
        """
        Adds a molecule to the route.

        Args:
            mol (str): The molecule to add.
            parent_id (int): The index of the parent molecule.
        """
        self.mols.append(mol)
        self.values.append(None)
        self.templates.append(None)
        self.parents.append(parent_id)
        self.children.append(None)

        self.children[parent_id].append(len(self.mols)-1)

    def set_value(self, mol, value):
        """
        Sets the value of a molecule in the route.

        Args:
            mol (str): The molecule to set the value for.
            value: The value to set.
        """
        assert mol in self.mols

        mol_id = self.mols.index(mol)
        self.values[mol_id] = value

    def add_reaction(self, mol, value, template, reactants, cost):
        """
        Adds a reaction to the route.

        Args:
            mol (str): The molecule to add the reaction to.
            value: The value associated with the reaction.
            template: The template associated with the reaction.
            reactants (list): A list of reactant molecules.
            cost (float): The cost of the reaction.
        """
        assert mol in self.mols

        self.total_cost += cost
        self.length += 1

        parent_id = self.mols.index(mol)
        self.values[parent_id] = value
        self.templates[parent_id] = template
        self.children[parent_id] = []
        self.costs[parent_id] = cost

        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def viz_route(self, viz_file):
        """
        Visualizes the route and saves it to a file.

        Args:
            viz_file (str): The file path to save the visualization.
        """
        G = Digraph('G', filename=viz_file)
        G.attr('node', shape='box')
        G.format = 'pdf'

        names = []
        for i in range(len(self.mols)):
            name = self.mols[i]
            # if self.templates[i] is not None:
            #     name += ' | %s' % self.templates[i]
            names.append(name)

        node_queue = Queue()
        node_queue.put((0,-1))   # target mol idx, and parent idx
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()

            if parent_idx >= 0:
                G.edge(names[parent_idx], names[idx], label='cost')

            if self.children[idx] is not None:
                for c in self.children[idx]:
                    node_queue.put((c, idx))

        G.render()

    def serialize_reaction(self, idx):
        """
        Serializes a reaction in the route.

        Args:
            idx (int): The index of the reaction.

        Returns:
            str: The serialized reaction.
        """
        s = self.mols[idx]
        if self.children[idx] is None:
            return s
        s += '>%.4f>' % np.exp(-self.costs[idx])
        s += self.mols[self.children[idx][0]]
        for i in range(1, len(self.children[idx])):
            s += '.'
            s += self.mols[self.children[idx][i]]

        return s

    def serialize(self):
        """
        Serializes the entire route.

        Returns:
            str: The serialized route.
        """
        s = self.serialize_reaction(0)
        for i in range(1, len(self.mols)):
            if self.children[i] is not None:
                s += '|'
                s += self.serialize_reaction(i)

        return s