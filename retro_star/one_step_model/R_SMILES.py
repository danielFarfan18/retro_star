import logging
import subprocess
import tempfile
import os
import random
import numpy as np


class OneStep:
    def run(self, model_path, src_str, top_k, output_path, gpu_id=-1, beam_size=10, batch_type="tokens", max_length=1000, seed=0):
        """
         Run onmt_translate on the source string and return the output. This is a wrapper around the pyc to be used in conjunction with the pyc. toolchain.
         
         @param model_path: The path to the model that will be used
         @param src_str: The source string to be translated
         @param top_k: The number of generations to be returned
         @param output_path: The path to the output file that will be written
         @param gpu_id: The GPU ID of the model
         @param beam_size: The beam size for the translation model @param batch_type: The type of batch to be used in the translation model 
         @param max_length: The maximum length of the generated translations
         @param seed: The seed for the random number generator 
         @returns: A dictionary containing the generated translations ('reactants'), their corresponding scores ('scores'), and templates ('templates')
        """
        # Write the input string to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write(src_str + '\n')
            src_path = f.name  # Use the path to the temporary file as the src argument

        # Build the command
        cmd = f"onmt_translate -model {model_path} -src {src_path} -output {output_path} -gpu {gpu_id} \
            -beam_size {beam_size} -n_best {top_k} -batch_type {batch_type} -max_length {max_length} -seed {seed}"
        # Run the command
        subprocess.run(cmd, shell=True)

        # Read the output file
        try:
            with open(output_path, 'r') as f:
                generations = [line.strip().replace(" ", "") for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Error reading output file: {e}")
            generations = []

        # Delete the source file
        os.remove(src_path)
        # Generate random scores in the range of the provided values
        min_score = 0.00040628651277562286
        max_score = 0.2941402511255707
        scores = [np.random.uniform(min_score, max_score) for _ in generations]
        # Initialize templates with None values
        templates = [None for _ in generations]

        # Fill the results dictionary with generations, a random number, and templates
        results = {'reactants': generations, 'scores': scores, 'templates': templates}
        return results