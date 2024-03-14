"""This script is used for retrosynthetic planning using a method called Retro* (Retro-Star).
Retrosynthetic planning is a process in chemistry where you start with a target molecule 
and plan backwards to find simpler starting molecules and the reactions needed to synthesize the target.
"""
import numpy as np
import torch
import random
import logging
import time
import pickle
import os
from retro_star.common import args, prepare_starting_molecules, prepare_mlp, \
    prepare_molstar_planner, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger

def retro_plan():
    # Set up the device for computation (CPU or GPU).
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    # Prepare the starting molecules.
    starting_mols = prepare_starting_molecules(args.starting_molecules)

    # Load the test routes (retrosynthetic plans) from a file.
    routes = pickle.load(open(args.test_routes, 'rb'))
    logging.info('%d routes extracted from %s loaded' % (len(routes), args.test_routes))

    # Prepare the MLP (Multi-Layer Perceptron) model for one-step retrosynthesis prediction.
    one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump)

    # Create a result folder if it doesn't exist.
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    # If use_value_fn is True, load a pre-trained ValueMLP model from a file and define a value function 
    # that uses this model to predict the value of a molecule. 
    # If use_value_fn is False, define a value function that always returns 0.
    if args.use_value_fn:
        model = ValueMLP(
            n_layers=args.n_layers,
            fp_dim=args.fp_dim,
            latent_dim=args.latent_dim,
            dropout_rate=0.1,
            device=device
        ).to(device)
        model_f = '%s/%s' % (args.save_folder, args.value_model)
        logging.info('Loading value nn from %s' % model_f)
        model.load_state_dict(torch.load(model_f,  map_location=device))
        model.eval()

        def value_fn(mol):
            fp = smiles_to_fp(mol, fp_dim=args.fp_dim).reshape(1,-1)
            fp = torch.FloatTensor(fp).to(device)
            v = model(fp).item()
            return v
    else:
        value_fn = lambda x: 0.

    # Prepare the Retro* planner with the one-step MLP, the value function, and the starting molecules.
    plan_handle = prepare_molstar_planner(
        one_step=one_step,
        value_fn=value_fn,
        starting_mols=starting_mols,
        expansion_topk=args.expansion_topk,
        iterations=args.iterations,
        viz=args.viz,
        viz_dir=args.viz_dir
    )

    # Initialize an empty dictionary to store the results.
    result = {
        'succ': [],
        'cumulated_time': [],
        'iter': [],
        'routes': [],
        'route_costs': [],
        'route_lens': []
    }
    num_targets = len(routes)
    t0 = time.time()
    for (i, route) in enumerate(routes):
        # For each route in the test routes, use the Retro* planner to plan the synthesis of the target molecule. 
        # Store the success status, cumulative time, number of iterations, planned route, route cost, and route length in the result dictionary.
        target_mol = route[0].split('>')[0]
        succ, msg = plan_handle(target_mol, i)

        result['succ'].append(succ)
        result['cumulated_time'].append(time.time() - t0)
        result['iter'].append(msg[1])
        result['routes'].append(msg[0])
        if succ:
            result['route_costs'].append(msg[0].total_cost)
            result['route_lens'].append(msg[0].length)
        else:
            result['route_costs'].append(None)
            result['route_lens'].append(None)

        tot_num = i + 1
        tot_succ = np.array(result['succ']).sum()
        avg_time = (time.time() - t0) * 1.0 / tot_num
        avg_iter = np.array(result['iter'], dtype=float).mean()
        logging.info('Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f' %
                     (tot_succ, tot_num, num_targets, avg_time, avg_iter))

    # After planning for all routes, save the result dictionary to a file.
    f = open(args.result_folder + '/plan.pkl', 'wb')
    pickle.dump(result, f)
    f.close()

if __name__ == '__main__':
    # If the script is run as the main program, set the random seed for numpy, torch, and the random module to ensure reproducibility, 
    # set up a logger, and call the retro_plan function.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('plan.log')

    retro_plan()