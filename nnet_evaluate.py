"""Evaluate models trained with nnet_train.py

Author: Daniel da Silva <daniel.e.dasilva@nasa.gov>
"""
import os

import h5py
import joblib
import numpy as np
from scipy.stats import linregress
from tensorflow.keras import backend

#from moms_numpy import idl_moments
from moms_fast import fast_moments

from nnet_train import (
    N_EN, N_EN_SHELLS,
    get_hidden_layer_sizes,
    load_model,
)
from utils import get_f1ct, load_test_data



def evaluate_r2(phase):
    """Apply neural networks trained with variable hidden layer sizes
    to test data and compute/store linear regressions between true and
    reconstructed moments.

    Args
      phase: mission phase
    """
    # Load test data
    # ------------------------------------------------------------------------
    print('Loading test data')

    test_data = load_test_data(phase)
    
    # Perform regressions for each hidden layer size
    # ------------------------------------------------------------------------    
    regressions_by_size = {}
    points_by_size = {}    
    sizes = get_hidden_layer_sizes()

    print('Hidden layer sizes: ' + str(sizes))
    
    for i, size in enumerate(sizes):
        print(f'Performing regressions for Hidden Layer Size = {size} '
              f'({i+1}/{len(sizes)})')
        print('-----------------------------------------------------')

        regressions_by_size[size], points_by_size[size] = (
            evaluate_r2_by_hidden_layer_size(phase, size, test_data)
        )
        
        print('-----------------------------------------------------')

    backend.clear_session()
        
    # Write to disk
    # ------------------------------------------------------------------------
    print('Writing output to disk')

    outname = (f'/mnt/efs/dasilva/compression-cfha/data/nnet_models'
               f'/hidden_layer_exp/{phase}/moments_stats.hdf')
    variables = regressions_by_size[sizes[0]].keys()
    
    hdf = h5py.File(outname, 'w')
    hdf['sizes'] = sizes

    for var in variables:    
        group = hdf.create_group(var)
        
        group['slope'] = np.array([
            regressions_by_size[size][var].slope for size in sizes
        ])
        group['intercept'] = np.array([
            regressions_by_size[size][var].intercept for size in sizes
        ])        
        group['r2'] = np.array([
            regressions_by_size[size][var].rvalue**2 for size in sizes
        ])
        group['points_true'] = np.array([
            points_by_size[size][var][0] for size in sizes
        ])
        group['points_recon'] = np.array([
            points_by_size[size][var][1] for size in sizes
        ])
    
    hdf.close()

    print('Done')

    
def evaluate_r2_by_hidden_layer_size(phase, hidden_layer_size, test_data):
    """Apply neural networks trained with a particular hidden layer size
    to test data and compute linear regressions between true / reconstructed
    moments.

    Args
      phase: mission phase
      hidden_layer_size: integer hidden layer size
      test_data: dictionary of data from load_test_data()
    Returns
      dictionary mapping moments variable name to scipy.stats.linregress
      return object.
    """
    # Load models for each energy index
    # ------------------------------------------------------------------------
    print('Loading models')
    models = {}

    for en_index in range(0, N_EN, N_EN_SHELLS):
        models[en_index] = load_model(phase, hidden_layer_size, en_index)

    n_items = test_data['counts'].shape[0]
    
    # Reconstruct
    # ------------------------------------------------------------------------
    print('Reconstructing data')
    counts_recon = np.zeros_like(test_data['counts'])

    for en_index in range(0, N_EN, N_EN_SHELLS):
        i, di = en_index, N_EN_SHELLS
        
        model_input = test_data['counts'][:, :, :, i:i+di]        
        model_output = models[i](model_input).numpy()
        
        counts_recon[:, :, :, i:i+di] = model_output

        # heuristic!
        for j in range(test_data['counts'].shape[0]):
            avg_orig = test_data['counts'][j, :, :, i:i+di].mean()
            avg_recon = counts_recon[j, :, :, i:i+di].mean()
    
            if avg_orig == 0:
                counts_recon[j, :, :, i:i+di] = 0
            elif avg_recon > 0:
                counts_recon[j, :, :, i:i+di] *=  avg_orig / avg_recon 
                
    # Calculate moments
    # ------------------------------------------------------------------------
    print(f'Calculating moments in Parallel (nprocs = {os.cpu_count()})')    

    f1ct = get_f1ct({phase: test_data}, [phase])
    tasks_true = []
    tasks_recon = []
    
    for i in range(n_items):
        if test_data['dist'][i].any():
            dist_orig = test_data['counts'][i] * f1ct
            dist_recon = counts_recon[i] * f1ct            
            tasks_true.append(joblib.delayed(fast_moments)(dist_orig))
            tasks_recon.append(joblib.delayed(fast_moments)(dist_recon))
        else:
            print('Skipping dist because all zeros..')
            
    moments_true = joblib.Parallel(n_jobs=-1)(tasks_true)
    moments_recon = joblib.Parallel(n_jobs=-1)(tasks_recon)
        
    # Perform regression on moments
    # ------------------------------------------------------------------------
    print('Performing Linear Regressions')
    variables = moments_true[0].keys()
    regressions = {}
    points = {}
    
    for var in variables:
        x = np.array([moms[var] for moms in moments_true])
        y = np.array([moms[var] for moms in moments_recon])

        mask = np.isfinite(x) & np.isfinite(y)

        if (~mask).any():
            print(f'Dropping {np.sum(~mask)} elements due to NaNs problem')

        regressions[var] = linregress(x[mask], y[mask])
        points[var] = (x, y)
        
    return regressions, points


if __name__ == '__main__':
    evaluate_r2('4A_dusk_flank')
    #evaluate_r2('4B_dayside')
    #evaluate_r2('4C_dawn_flank')
    #evaluate_r2('4D_tail')
    #evaluate_r2('all')
    
