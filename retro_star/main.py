import torch
import logging
import time
# Importing necessary modules from the project
from common import *
from model import ValueMLP
from utils import setup_logger

import os
from common.prepare_utils import prepare_single_step_model

# Getting the directory path of the current file
dirpath = os.path.dirname(os.path.abspath(__file__))

# Defining the RSPlanner class
class RSPlanner:
    # The constructor for the RSPlanner class
    def __init__(self,
                 gpu=-1,  # The GPU device number to use. If -1, CPU is used.
                 expansion_topk=50,  # The number of top molecules to consider for expansion
                 iterations=500,  # The number of iterations to perform
                 use_value_fn=True,  # Whether to use a value function to guide the search
                 starting_molecules=dirpath+'/dataset/origin_dict.csv',  # The path to the file containing the starting molecules
                 mlp_model_dump=dirpath+'/one_step_model/R_SMILES.pt',  # The path to the file containing the single step model (R-SIMLES)
                 save_folder=dirpath+'/saved_models',  # The folder to save the models
                 value_model='best_epoch_final_4.pt',  # The name of the value model file
                 fp_dim=2048,  # The dimension of the fingerprint
                 viz=True,  # Whether to visualize the planning process
                 viz_dir='viz'):  # The directory to save the visualizations
        """
        Initializes the RetroStar object.

        Parameters:
        - gpu (int): The GPU device number to use. If -1, CPU is used.
        - expansion_topk (int): The number of top molecules to consider for expansion.
        - iterations (int): The number of iterations to perform.
        - use_value_fn (bool): Whether to use a value function to guide the search.
        - starting_molecules (str): The path to the file containing the starting molecules.
        - mlp_model_dump (str): The path to the file containing the single step model (R-SIMLES).
        - save_folder (str): The folder to save the models.
        - value_model (str): The name of the value model file.
        - fp_dim (int): The dimension of the fingerprint.
        - viz (bool): Whether to visualize the planning process.
        - viz_dir (str): The directory to save the visualizations.
        """
        setup_logger()
        # Setting the device for torch
        device = torch.device('cuda:%d' % gpu if torch.cuda.is_available() and gpu >= 0 else 'cpu')
        # Preparing the starting molecules
        starting_mols = prepare_starting_molecules(starting_molecules)

        # load single step model (R-SIMLES)
        one_step = prepare_single_step_model(mlp_model_dump, device)

        # If use_value_fn is True, load the model and define the value function
        if use_value_fn:
            model = ValueMLP(
                n_layers=1,
                fp_dim=fp_dim,
                latent_dim=128,
                dropout_rate=0.1,
                device=device
            ).to(device)
            model_f = '%s/%s' % (save_folder, value_model)
            logging.info('Loading value nn from %s' % model_f)
            model.load_state_dict(torch.load(model_f, map_location=device))
            model.eval()

            def value_fn(mol):
                fp = smiles_to_fp(mol, fp_dim=fp_dim).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
        else:
            # If use_value_fn is False, define a dummy value function
            value_fn = lambda x: 0.

        # Preparing the planner
        self.plan_handle = prepare_molstar_planner(
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
        )

    # The plan method for the RSPlanner class
    def plan(self, target_mol):  # The target molecule to plan for
        t0 = time.time()
        succ, msg = self.plan_handle(target_mol)

        # If the plan is successful, return the result
        if succ:
            result = {
                'succ': succ,  # Whether the plan was successful
                'time': time.time() - t0,  # The time taken for the plan
                'iter': msg[1],  # The number of iterations performed
                'routes': msg[0].serialize(),  # The routes found
                'route_cost': msg[0].total_cost,  # The total cost of the route
                'route_len': msg[0].length  # The length of the route
            }
            return result

        else:
            # If the plan is not successful, log the message and return None
            logging.info('Synthesis path for %s not found. Please try increasing '
                         'the number of iterations.' % target_mol)
            return None


# If the script is run directly, create an instance of RSPlanner and call the plan method
if __name__ == '__main__':
    planner = RSPlanner(
        gpu=0,
        use_value_fn=True,
        iterations=100,
        expansion_topk=50
    )

    result = planner.plan('N[C@H]1CC[C@H]1c1ccc(Cl)cc1')
    print(result)

    #result = planner.plan('CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C')
    #print(result)

    #result = planner.plan('CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1')
    #print(result)