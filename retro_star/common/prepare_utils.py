import torch
import pandas as pd
import logging
from alg import molstar
from one_step_model.R_SMILES import OneStep


def prepare_starting_molecules(filename):
    """
    Load and prepare starting molecules from a csv or pkl file.

    Args:
        filename (str): The path to the csv or pkl file containing the starting molecules.

    Returns:
        set: A set of starting molecules.

    Raises:
        AssertionError: If the file extension is not csv or pkl.

    """
    logging.info('Loading starting molecules from %s' % filename)

    # Load the starting molecule from a csv or pkl file.
    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols


def prepare_single_step_model(model_dump, device):
    """
    Loads and returns a trained single step model.
    
    This is a convenience function for preparing a model for training and evaluation.
    
    Args:
        model_dump (str): Path to the model to load.
        device (str): Device to load the model on. Default is the device used by TensorFlow.
    
    Returns:
        torch.nn.Module: A trained and evaluated model ready to be used as a train_fn or eval_fn.
    """
    logging.info('Loading trained single step model from %s' % model_dump)
    one_step = torch.load(model_dump, map_location=device)
    return one_step


def prepare_molstar_planner(one_step, value_fn, starting_mols, expansion_topk,
                            iterations, viz=False, viz_dir=None):
    """
    Prepares the Mol* planner by defining function handles for the expansion step and the Mol* planner.

    Args:
        one_step (OneStep): An instance of the OneStep class.
        value_fn (function): The value function used by the Mol* planner.
        starting_mols (list): List of starting molecules.
        expansion_topk (int): Number of top-k expansions to consider.
        iterations (int): Number of iterations for the Mol* planner.
        viz (bool, optional): Whether to visualize the planning process. Defaults to False.
        viz_dir (str, optional): Directory to save the visualization files. Defaults to None.

    Returns:
        function: Function handle to the Mol* planner.
    """
    # Initialize an instance of the OneStep class
    one_step = OneStep()
    # Define the path to the model
    model_path = "one_step_model/R_SMILES.pt"
    # Define the path to the output file
    output_path = "exp/outputs.txt"

    # Define a function handle for the expansion step
    expansion_handle = lambda x: one_step.run(model_path=model_path, src_str=x, top_k=expansion_topk, output_path=output_path)

    # Define a function handle for the Mol* planner
    plan_handle = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir
    )
    # Return the function handle to the Mol* planner
    return plan_handle