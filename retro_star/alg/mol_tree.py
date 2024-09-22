import numpy as np
from queue import Queue
import logging
import networkx as nx
from graphviz import Digraph
from alg.mol_node import MolNode
from alg.reaction_node import ReactionNode
from alg.syn_route import SynRoute


class MolTree:

    def __init__(self, target_mol, known_mols, value_fn, zero_known_value=True):
        """
        Initializes a MolTree object.

        Args:
            target_mol (Molecule): The target molecule to synthesize.
            known_mols (list): List of known starting molecules.
            value_fn (function): Function to calculate the value of a molecule.
            zero_known_value (bool, optional): Whether to set the value of known molecules to zero. Defaults to True.

        Attributes:
            target_mol (Molecule): The target molecule to synthesize.
            known_mols (list): List of known starting molecules.
            value_fn (function): Function to calculate the value of a molecule.
            zero_known_value (bool): Whether to set the value of known molecules to zero.
            mol_nodes (list): List of molecular nodes in the tree.
            reaction_nodes (list): List of reaction nodes in the tree.
            root (MolNode): The root node of the tree.
            succ (bool): Indicates whether a synthesis route has been found.
            search_status (int): The search status of the tree.

        Methods:
            _add_mol_node(self, mol, parent): Adds a molecular node to the tree.
            _add_reaction_and_mol_nodes(self, cost, mols, parent, template, ancestors): Adds a reaction node and its associated molecular nodes to the tree.
            expand(self, mol_node, reactant_lists, costs, templates): Expands the tree by adding reaction nodes and their associated molecular nodes.
            get_best_route(self): Returns the best synthesis route found in the tree.
            viz_search_tree(self, viz_file): Visualizes the search tree and saves it to a file.
        """
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.value_fn = value_fn
        self.zero_known_value = zero_known_value
        self.mol_nodes = []
        self.reaction_nodes = []

        self.root = self._add_mol_node(target_mol, None)
        self.succ = target_mol in known_mols
        self.search_status = 0

        if self.succ:
            logging.info('Synthesis route found: target in starting molecules')
            
    def _add_mol_node(self, mol, parent):
        """
        Add a new molecule node to the molecular tree.

        Args:
            mol (Molecule): The molecule to be added as a node.
            parent (MolNode): The parent node of the molecule.

        Returns:
            MolNode: The newly created molecule node.

        """
        # Check if the molecule is known
        is_known = mol in self.known_mols

        # Calculate the initial value of the molecule
        init_value = self.value_fn(mol)

        # Create a new molecule node
        mol_node = MolNode(
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known,
            zero_known_value=self.zero_known_value
        )
        # Add the new molecule node to the list of molecule nodes
        self.mol_nodes.append(mol_node)
        # Assign an ID to the molecule node
        mol_node.id = len(self.mol_nodes)

        return mol_node

    def _add_reaction_and_mol_nodes(self, cost, mols, parent, template, ancestors):
        """
        Adds a new reaction node and corresponding molecule nodes to the molecular tree.

        Args:
            cost (float): The cost associated with the reaction.
            mols (list): A list of molecules involved in the reaction.
            parent (ReactionNode): The parent reaction node.
            template (Template): The template associated with the reaction.
            ancestors (set): A set of molecules that are ancestors of the current reaction.

        Returns:
            ReactionNode: The newly created reaction node.

        Raises:
            AssertionError: If the cost is negative.

        """
        # Ensure the cost is non-negative
        assert cost >= 0

        # Check if any of the molecules are ancestors to avoid cycles
        for mol in mols:
            if mol in ancestors:
                return

        # Create a new reaction node
        reaction_node = ReactionNode(parent, cost, template)
        # Add a new molecule node for each molecule in the reaction
        for mol in mols:
            self._add_mol_node(mol, reaction_node)
        # Initialize the values of the reaction node
        reaction_node.init_values()
        # Add the new reaction node to the list of reaction nodes
        self.reaction_nodes.append(reaction_node)
        # Assign an ID to the reaction node
        reaction_node.id = len(self.reaction_nodes)

        return reaction_node

    def expand(self, mol_node, reactant_lists, costs, templates):
            """
            Expands the given molecule node by adding new reaction nodes and their associated molecule nodes.

            Args:
                mol_node (MolNode): The molecule node to expand.
                reactant_lists (list): A list of reactant lists for each possible reaction.
                costs (list): A list of costs for each possible reaction.
                templates (list): A list of templates for each possible reaction.

            Returns:
                bool: True if a synthesis route has been found, False otherwise.
            """
            # Ensure the molecule node is not known and has no children
            assert not mol_node.is_known and not mol_node.children

            # If there are no expansion results
            if costs is None:
                # Ensure the molecule node's value is infinity
                assert mol_node.init_values(no_child=True) == np.inf
                # If the molecule node has a parent, update the parent's value
                if mol_node.parent:
                    mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
                return self.succ

            # Ensure the molecule node is open
            assert mol_node.open
            # Get the ancestors of the molecule node
            ancestors = mol_node.get_ancestors()
            # Add a new reaction node and its associated molecule nodes for each possible reaction
            for i in range(len(costs)):
                self._add_reaction_and_mol_nodes(costs[i], reactant_lists[i],
                                                 mol_node, templates[i], ancestors)

            # If there are no valid expansion results
            if len(mol_node.children) == 0:
                # Ensure the molecule node's value is infinity
                assert mol_node.init_values(no_child=True) == np.inf
                # If the molecule node has a parent, update the parent's value
                if mol_node.parent:
                    mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
                return self.succ

            # Calculate the change in value of the molecule node
            v_delta = mol_node.init_values()
            # If the molecule node has a parent, update the parent's value
            if mol_node.parent:
                mol_node.parent.backup(v_delta, from_mol=mol_node.mol)

            # If a synthesis route has been found
            if not self.succ and self.root.succ:
                logging.info('Synthesis route found!')
                self.succ = True

            return self.succ

    def get_best_route(self):
        """
        Returns the best synthesis route for the molecule tree.

        If no synthesis route has been found, returns None.

        Returns:
            SynRoute: The best synthesis route for the molecule tree.
        """
        # If no synthesis route has been found, return None
        if not self.succ:
            return None

        # Create a new synthesis route
        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status
        )

        # Create a queue of molecule nodes and add the root to it
        mol_queue = Queue()
        mol_queue.put(self.root)
        while not mol_queue.empty():
            mol = mol_queue.get()
            # If the molecule is known, set its value in the synthesis route
            if mol.is_known:
                syn_route.set_value(mol.mol, mol.succ_value)
                continue

            # Find the best reaction for the molecule
            best_reaction = None
            for reaction in mol.children:
                if reaction.succ:
                    if best_reaction is None or \
                            reaction.succ_value < best_reaction.succ_value:
                        best_reaction = reaction
            # Ensure the best reaction's value is the same as the molecule's value
            assert best_reaction.succ_value == mol.succ_value

            # Get the reactants of the best reaction
            reactants = []
            for reactant in best_reaction.children:
                mol_queue.put(reactant)
                reactants.append(reactant.mol)

            # Add the best reaction to the synthesis route
            syn_route.add_reaction(
                mol=mol.mol,
                value=mol.succ_value,
                template=best_reaction.template,
                reactants=reactants,
                cost=best_reaction.cost
            )

        return syn_route

    def viz_search_tree(self, viz_file):
        """
        Visualizes the search tree using Graphviz and saves it to a file.

        Parameters:
        - viz_file (str): The file path to save the visualization.

        Returns:
        - None
        """
        # Create a new directed graph
        G = Digraph('G', filename=viz_file)
        # Set the direction of the graph from left to right
        G.attr(rankdir='LR')
        # Set the shape of the nodes to box
        G.attr('node', shape='box')
        # Set the format of the graph to PDF
        G.format = 'pdf'

        # Create a queue of nodes and add the root to it
        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()

            # Set the color of the node based on its status
            if node.open:
                color = 'lightgrey'
            else:
                color = 'aquamarine'

            # Set the shape of the node based on its type
            if hasattr(node, 'mol'):
                shape = 'box'
            else:
                shape = 'rarrow'

            # If the node is successful, set its color
            if node.succ:
                color = 'lightblue'
                if hasattr(node, 'mol') and node.is_known:
                    color = 'lightyellow'

            # Add the node to the graph
            G.node(node.serialize(), shape=shape, color=color, style='filled')

            # Set the label of the edge based on the parent's type
            label = ''
            if hasattr(parent, 'mol'):
                label = '%.3f' % node.cost
            # If the node has a parent, add an edge from the parent to the node
            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)

            # If the node has children, add them to the queue
            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        # Render the graph
        G.render()