import numpy as np
import logging


class ReactionNode:
    def __init__(self, parent, cost, template):
        """
        Initializes a ReactionNode object.

        Args:
            parent (ReactionNode): The parent node of the current node.
            cost (float): The cost associated with the current node.
            template: The template associated with the current node.
        """
        self.parent = parent
        
        self.depth = self.parent.depth + 1
        self.id = -1

        self.cost = cost
        self.template = template
        self.children = []
        self.value = None   # [V(m | subtree_m) for m in children].sum() + cost
        self.succ_value = np.inf    # total cost for existing solution
        self.target_value = None    # V_target(self | whole tree)
        self.succ = None    # successfully found a valid synthesis route
        self.open = True    # before expansion: True, after expansion: False
        parent.children.append(self)

    def v_self(self):
        """
        Calculates the value of the current node.

        Returns:
            float: The value of the current node.
        """
        return self.value

    def v_target(self):
        """
        Calculates the target value of the current node.

        Returns:
            float: The target value of the current node.
        """
        return self.target_value

    def init_values(self):
        """
        Initializes the values of the current node and its children.
        """
        assert self.open

        self.value = self.cost
        self.succ = True
        for mol in self.children:
            self.value += mol.value
            self.succ &= mol.succ

        if self.succ:
            self.succ_value = self.cost
            for mol in self.children:
                self.succ_value += mol.succ_value

        self.target_value = self.parent.v_target() - self.parent.v_self() + \
                            self.value
        self.open = False

    def backup(self, v_delta, from_mol=None):
        """
        Backs up the changes made to the current node and propagates the changes to its parent.

        Args:
            v_delta (float): The change in value.
            from_mol: The molecule from which the change originated.

        Returns:
            ReactionNode: The parent node.
        """
        self.value += v_delta
        self.target_value += v_delta

        self.succ = True
        for mol in self.children:
            self.succ &= mol.succ

        if self.succ:
            self.succ_value = self.cost
            for mol in self.children:
                self.succ_value += mol.succ_value

        if v_delta != 0:
            assert from_mol
            self.propagate(v_delta, exclude=from_mol)

        return self.parent.backup(self.succ)

    def propagate(self, v_delta, exclude=None):
        """
        Propagates the changes made to the current node to its children.

        Args:
            v_delta (float): The change in value.
            exclude: The molecule to exclude from propagation.
        """
        if exclude is None:
            self.target_value += v_delta

        for child in self.children:
            if exclude is None or child.mol != exclude:
                for grandchild in child.children:
                    grandchild.propagate(v_delta)

    def serialize(self):
        """
        Serializes the current node.

        Returns:
            str: The serialized representation of the current node.
        """
        return '%d' % (self.id)
        # return '%d | value %.2f | target %.2f' % \
        #        (self.id, self.v_self(), self.v_target())