import numpy as np
import logging


class MolNode:
    """
    Represents a node in a molecule tree.

    Attributes:
        mol: The molecule associated with the node.
        pred_value: The predicted value of the node.
        value: The current value of the node.
        succ_value: The total cost for existing solution.
        parent: The parent node of the current node.
        id: The ID of the node.
        depth: The depth of the node in the tree.
        is_known: A boolean indicating if the node is known.
        children: The children nodes of the current node.
        succ: A boolean indicating if the node is a successor.
        open: A boolean indicating if the node is open for expansion.
    """

    def __init__(self, mol, init_value, parent=None, is_known=False,
                 zero_known_value=True):
        """
        Initializes a MolNode object.

        Args:
            mol: The molecule associated with the node.
            init_value: The initial value of the node.
            parent: The parent node of the current node. Defaults to None.
            is_known: A boolean indicating if the node is known. Defaults to False.
            zero_known_value: A boolean indicating if the known value should be set to zero. Defaults to True.
        """
        self.mol = mol
        self.pred_value = init_value
        self.value = init_value
        self.succ_value = np.inf    # total cost for existing solution
        self.parent = parent

        self.id = -1
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth

        self.is_known = is_known
        self.children = []
        self.succ = is_known
        self.open = True    # before expansion: True, after expansion: False
        if is_known:
            self.open = False
            if zero_known_value:
                self.value = 0
            self.succ_value = self.value

        if parent is not None:
            parent.children.append(self)

    def v_self(self):
        """
        Returns the value of the current node.
        """
        return self.value

    def v_target(self):
        """
        Returns V_target(self | whole tree).
        """
        if self.parent is None:
            return self.value
        else:
            return self.parent.v_target()

    def init_values(self, no_child=False):
        """
        Initializes the values of the node and its children.

        Args:
            no_child: A boolean indicating if the node has no children. Defaults to False.

        Returns:
            The change in value (v_delta) of the node.
        """
        assert self.open and (no_child or self.children)

        new_value = np.inf
        self.succ = False
        for reaction in self.children:
            new_value = np.min((new_value, reaction.v_self()))
            self.succ |= reaction.succ

        v_delta = new_value - self.value
        self.value = new_value

        if self.succ:
            for reaction in self.children:
                self.succ_value = np.min((self.succ_value,
                                          reaction.succ_value))

        self.open = False

        return v_delta

    def backup(self, succ):
        """
        Backs up the node's value and successor information to its parent.

        Args:
            succ: A boolean indicating if the node is a successor.

        Returns:
            The change in value (v_delta) of the node.
        """
        assert not self.is_known

        new_value = np.inf
        for reaction in self.children:
            new_value = np.min((new_value, reaction.v_self()))
        new_succ = self.succ | succ
        updated = (self.value != new_value) or (self.succ != new_succ)

        new_succ_value = np.inf
        if new_succ:
            for reaction in self.children:
                new_succ_value = np.min((new_succ_value, reaction.succ_value))
            updated = updated or (self.succ_value != new_succ_value)

        v_delta = new_value - self.value
        self.value = new_value
        self.succ = new_succ
        self.succ_value = new_succ_value

        if updated and self.parent:
            return self.parent.backup(v_delta, from_mol=self.mol)

    def serialize(self):
        """
        Serializes the node into a string representation.

        Returns:
            The serialized string representation of the node.
        """
        text = '%d | %s' % (self.id, self.mol)
        # text = '%d | %s | pred %.2f | value %.2f | target %.2f' % \
        #        (self.id, self.mol, self.pred_value, self.v_self(),
        #         self.v_target())
        return text

    def get_ancestors(self):
        """
        Returns a set of all the ancestors of the node.

        Returns:
            A set containing all the ancestors of the node.
        """
        if self.parent is None:
            return {self.mol}

        ancestors = self.parent.parent.get_ancestors()
        ancestors.add(self.mol)
        return ancestors