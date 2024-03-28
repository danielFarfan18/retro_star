"""
This script defines the RSPlanner class for retrosynthetic planning in chemistry. 
Retrosynthetic planning is a process where you start with a target molecule and plan backwards 
to find simpler starting molecules and the reactions needed to synthesize the target. 

The RSPlanner class uses a Multi-Layer Perceptron (MLP) model for one-step retrosynthesis prediction 
and optionally a ValueMLP model as a value function to guide the search. 

The plan method of the RSPlanner class performs the retrosynthetic planning for a given target molecule 
and returns the result. 

If the script is run directly, it creates an instance of RSPlanner and calls the plan method for 
three different target molecules.
"""
import torch
import logging
import time
# Importing necessary modules from the project
from retro_star.common import *
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from common.prepare_utils import prepare_single_step_model
import os
# Getting the directory path of the current file
dirpath = os.path.dirname(os.path.abspath(__file__))

# Defining the RSPlanner class
class RSPlanner:
    # The constructor for the RSPlanner class
    def __init__(self,
                 gpu=-1,  # The GPU device number to use. If -1, CPU is used.
                 expansion_topk=50,  # The number of top molecules to consider for expansion
                 iterations=500,  # The number of iterations to perform
                 use_value_fn=False,  # Whether to use a value function to guide the search
                 starting_molecules=dirpath+'/dataset/origin_dict.csv',  # The path to the file containing the starting molecules
                 mlp_templates=dirpath+'/one_step_model/template_rules_1.dat',  # The path to the file containing the MLP templates
                 mlp_model_dump=dirpath+'/one_step_model/saved_rollout_state_1_2048.ckpt',  # The path to the file containing the MLP model dump
                 save_folder=dirpath+'/saved_models',  # The folder to save the models
                 value_model='best_epoch_final_4.pt',  # The name of the value model file
                 fp_dim=2048,  # The dimension of the fingerprint
                 viz=False,  # Whether to visualize the planning process
                 viz_dir='viz'):  # The directory to save the visualizations

        # Setting up the logger
        setup_logger()
        # Setting the device for torch
        device = torch.device('cuda:%d' % gpu if gpu >= 0 else 'cpu')
        # Preparing the starting molecules
        starting_mols = prepare_starting_molecules(starting_molecules)

        # Preparing the MLP
        one_step = prepare_single_step_model(mlp_model_dump)

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

    result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
    print(result)

    result = planner.plan('CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C')
    print(result)

    result = planner.plan('CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1')
    print(result)