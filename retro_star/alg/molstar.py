import os
import numpy as np
import logging
from alg.mol_tree import MolTree


def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn,
            iterations, viz=False, viz_dir=None):
    """
    Perform a molecular search using the MolTree algorithm.

    Args:
        target_mol (str): The target molecule to search for.
        target_mol_id (int): The ID of the target molecule.
        starting_mols (list): A list of starting molecules.
        expand_fn (function): A function that expands a given molecule.
        value_fn (function): A function that calculates the value of a molecule.
        iterations (int): The maximum number of iterations to perform.
        viz (bool, optional): Whether to visualize the search process. Defaults to False.
        viz_dir (str, optional): The directory to save the visualization files. Defaults to None.

    Returns:
        tuple: A tuple containing the success status of the search and the best route found.
    """
    # Initialize a molecular tree with the target molecule, known molecules, and a value function
    mol_tree = MolTree(
            target_mol=target_mol,
            known_mols=starting_mols,
            value_fn=value_fn
        )

    # Initialize iteration counter and best route
    i = -1
    best_route = None

    # If the molecular tree is not successful
    if not mol_tree.succ:
        # Iterate for a specified number of iterations
        for i in range(iterations):
            scores = []
            # For each molecule node in the molecular tree
            for m in mol_tree.mol_nodes:
                # If the node is open, append its target value to the scores
                if m.open:
                    scores.append(m.v_target())
                # If the node is not open, append infinity to the scores
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            # If the minimum score is infinity, log that there are no open nodes and break the loop
            if np.min(scores) == np.inf:
                logging.info('No open nodes!')
                break

            # Set the metric to the scores
            metric = scores

            # Set the search status of the molecular tree to the minimum metric
            mol_tree.search_status = np.min(metric)
            # Set the next molecule to the molecule node with the minimum metric
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            # Expand the next molecule
            result = expand_fn(m_next.mol)

            # If the result is not None and there are scores
            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                # If there are templates in the result keys, set templates to them
                if 'templates' in result.keys():
                    templates = result['templates']
                # Otherwise, set templates to the template in the result
                else:
                    templates = result['template']

                reactant_lists = []
                # For each score, create a list of reactants and append it to the reactant lists
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                # Expand the molecular tree with the next molecule, reactant lists, costs, and templates
                succ = mol_tree.expand(m_next, reactant_lists, costs, templates)

                # If successful, break the loop
                if succ:
                    break

                # If the success value of the root of the molecular tree is less than or equal to the search status, break the loop
                if mol_tree.root.succ_value <= mol_tree.search_status:
                    break

            # If the result is None or there are no scores, expand the molecular tree with the next molecule and None for the reactant lists, costs, and templates
            else:
                mol_tree.expand(m_next, None, None, None)
                logging.info('Expansion fails on %s!' % m_next.mol)

        # Log the final search status, success value, and iteration
        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    # If the molecular tree is successful, get the best route
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    # If visualization is enabled
    if viz:
        # If the visualization directory does not exist, create it
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        # If the molecular tree is successful
        if mol_tree.succ:
            # If the best route is optimal, set the file path to include 'optimal'
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            # Otherwise, set the file path to not include 'optimal'
            else:
                f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            # Visualize the best route
            best_route.viz_route(f)

        # Set the file path for the search tree
        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        # Visualize the search tree
        mol_tree.viz_search_tree(f)

    # Return whether the molecular tree is successful and the best route and iteration
    return mol_tree.succ, (best_route, i+1)