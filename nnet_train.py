"""Code for the neural network experiment.

Author: Daniel da Silva <daniel.e.dasilva@nasa.gov>
"""
import os
import random
import shutil

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, layers, losses, optimizers
from tensorflow.keras.models import Model

# Skymap dimensions
N_EN = 32
N_PHI = 32
N_THETA = 16

# Number of energy shells per neural network application. Must divide
# cleanly into N_EN
N_EN_SHELLS = 2


class AutoEncoder(Model):
    """Single-hidden-layer auto-encoder model, with tunable
    hidden layer size.       
    """
    def __init__(self, hidden_layer_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(hidden_layer_size, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(N_PHI * N_THETA * N_EN_SHELLS, activation='relu'),
            layers.Reshape((N_PHI, N_THETA, N_EN_SHELLS))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_models(phase):
    """End-to-end function to train models."""
    # Setup output directory
    # ------------------------------------------------------------------------
    outpath = (f'/mnt/efs/dasilva/compression-cfha/data/nnet_models'
               f'/hidden_layer_exp/{phase}/')

    if os.path.exists(outpath):
        shutil.rmtree(outpath)        

    os.makedirs(outpath)

    # Train models
    # ------------------------------------------------------------------------
    for en_index in range(0, N_EN, N_EN_SHELLS):
        train_models_per_en_index(phase, en_index, outpath)


def train_models_per_en_index(phase, en_index, outpath):
    """Helper function to train models for one energy index

    Args
      phase: Mission phase to train models for
      en_index: Energy index to train models for
      outpath: Directory to place output data
    
    """
    # Loading training and validation data 
    # ------------------------------------------------------------------------
    train_file = '/mnt/efs/dasilva/compression-cfha/data/samples_train_n=50000_nosw.hdf'
    test_file = '/mnt/efs/dasilva/compression-cfha/data/samples_test_n=10000_nosw.hdf'
    X_train = load_model_inputs(phase, train_file, en_index)
    X_test = load_model_inputs(phase, test_file, en_index)

    # Train each model
    # ------------------------------------------------------------------------
    models = get_experiment_models(en_index)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                                restore_best_weights=True)
      
    for i, (hidden_layer_size, model) in enumerate(models.items()):
        print(f'Training model for Energy Index {en_index}, model '
              f'{i+1}/{len(models)} with hidden layer '
              f'size {hidden_layer_size}')

        # Train the model
        model.fit(x=X_train[:50_000], y=X_train[:50_000],
                  #batch_size=50,
                  batch_size=50,
                  epochs=50,
                  callbacks=[callback],
                  validation_data=(X_test[:5_000], X_test[:5_000])
                  )
        
        # Save the model
        outname = os.path.join(
            outpath, f'model_{phase}_{hidden_layer_size:06d}_EN{en_index:02d}')
        model.save(outname)

        backend.clear_session()
        
    return models


def get_file_num_samples(phase, file_path):
    """Get exact number of samples in a file. The filename is an approximation.

    Returns
      integer number of samples
    """
    hdf = h5py.File(file_path, 'r')
    n_items = hdf[phase]['counts'].shape[0]
    hdf.close()

    return n_items


def load_model_inputs(phase, file_path, en_index, _cache={}):
    """Gets training/test data from disk as a single array. 
 
    Args
      phase: phase of mission
      file_path: Path to train/test data
      en_index: energy index to retrieve
    Returns
      X: numpy array
    """
    # Load
    print(f'Loading data from {file_path}')
    hdf = h5py.File(file_path, 'r')

    if phase == 'all':
        X = []
        for p in hdf:
            X.extend(hdf[p]['counts'][:, :, :, en_index:en_index+N_EN_SHELLS])
        X = np.array(X)
    else:
        X = hdf[phase]['counts'][:, :, :, en_index:en_index+N_EN_SHELLS]

    hdf.close()
    
    # Drop all zero skymaps
    X = X[X.any(axis=(1, 2, 3))]

    # Shuffle
    X = list(X)
    random.shuffle(X)
    X = np.array(X)
    
    return X


def get_hidden_layer_sizes():
    """Get list of hidden layer sizes used in experiment.

    To be used with the AutoEncoder class.

    Returns
      List of integer hidden layer sizes
    """
    # Version 001
    #max_size = int(1.25 * N_EN_SHELLS * N_PHI * N_THETA)
    #return list(range(50, max_size, 50))
    
    # Version 002
    max_size = int(1.25 * N_EN_SHELLS * N_PHI * N_THETA)

    sizes = []
    sizes.extend(range(1, 50, 5))
    sizes.extend(range(50, max_size, 50))
    #sizes.extend(range(4, 75, 8))
    #sizes.extend(range(75, max_size, 50))

    return sizes


def get_experiment_models(en_index):
    """Get models varying over hidden layer size to be used in
    experiment.

    Returns
      List of AutoEncoder models (with .compile() ran). Keys are
      hidden_layer_size.
    """
    models = {}

    for size in get_hidden_layer_sizes():
        models[size] = AutoEncoder(size)
        models[size].compile(
            optimizer=optimizers.Adam(),
            loss=losses.MeanSquaredError()
        )

    return models


def load_model(phase, hidden_layer_size, en_index, outpath=None):
    """Load model from disk.
    
    Args
      phase: phase of mission
      hidden_layer_size: hidden layer size of model
      en_index: energy index of model
    Returns
      AutoEncoder model
    """
    if outpath is None:
        outpath = (f'/mnt/efs/dasilva/compression-cfha/data/nnet_models'
                   f'/hidden_layer_exp/{phase}/')
    outname = os.path.join(
        outpath, f'model_{phase}_{hidden_layer_size:06d}_EN{en_index:02d}')

    model = tf.saved_model.load(outname)

    return model

    
if __name__ == '__main__':
    train_models('4A_dusk_flank')
    #train_models('4B_dayside')
    #train_models('4C_dawn_flank')
    #train_models('4D_tail')
    #train_models('all')
    
