#!/bin/env python

import argparse
from dataclasses import dataclass
import gzip
import os
import warnings

import bitstring
import numpy as np
import progressbar
from spacepy import pycdf

from nnet_train import (
    N_EN, N_PHI, N_THETA,
    load_model,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@dataclass
class CdfData:
    """Holds CDF data read from disk"""
    epoch: np.array
    counts: np.array


def get_cdf_data(mms_cdf_file):
    """"Loads CDF data from disk.

    Args
      mms_cdf_file: string path to mms cdf dist file
    Returns
      instance of CdfData class with access to data through attributes
    """
    cdf = pycdf.CDF(mms_cdf_file)

    for i in range(1, 5):
        if f'mms{i}' in mms_cdf_file:
            key = f'mms{i}'
            break

    dist = cdf[f'{key}_dis_dist_brst'][:]
    dist_err = cdf[f'{key}_dis_disterr_brst'][:]
    epoch = cdf['Epoch'][:]
    ntime = epoch.size
    counts = np.zeros((ntime, N_PHI, N_THETA, N_PHI))

    for i in range(ntime):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            tmp_counts = np.square(dist[i] / dist_err[i])
        tmp_counts[np.isnan(tmp_counts)] = 0
        tmp_counts = np.rint(tmp_counts)
        counts[i] = tmp_counts
    
    cdf.close()

    return CdfData(epoch=epoch, counts=counts)
    

def main():
    # Parse command line arguments
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('mms_cdf_file')
    parser.add_argument('out_file')
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-en-shells', type=int, required=True)
    parser.add_argument('--hidden-layer-size', type=int, required=True)
    parser.add_argument('--mantissa-bits', type=int, default=4)
    
    args = parser.parse_args()
    
    # Load data from CDF file
    # ------------------------------------------------------------------------
    cdf_data = get_cdf_data(args.mms_cdf_file)
    
    # Determine header
    # ------------------------------------------------------------------------
    header = bitstring.BitArray(uint=cdf_data.epoch.size, length=32)

    means = np.zeros((cdf_data.epoch.size, N_EN // args.n_en_shells),
                     dtype=np.float16)
    
    for en_index in range(N_EN // args.n_en_shells):
        i = en_index * args.n_en_shells
        di = args.n_en_shells
        means[:, en_index] = (
            cdf_data.counts[:, :, :, i:i+di].mean(axis=(1, 2, 3))
        )

    header.append(bitstring.BitArray(bytes=means.tobytes()))

    header_bytes = header.bytes
    
    # Load models for each energy index
    # ------------------------------------------------------------------------
    print('Loading models')
    model_name = args.model.split('.')[0]
    outpath = (f'/mnt/efs/dasilva/compression-cfha/data/nnet_models'
               f'/hidden_layer_exp/{args.model}/')

    models = {}

    for en_index in range(0, N_EN, args.n_en_shells):
        models[en_index] = load_model(model_name, args.hidden_layer_size,
                                      en_index, outpath=outpath)

    # Convert counts to latent representation
    # ------------------------------------------------------------------------
    latent_shape = (
        cdf_data.epoch.size, N_EN // args.n_en_shells, args.hidden_layer_size
    )
    latent = np.zeros(latent_shape, dtype=np.float16)
    
    for en_index in range(N_EN // args.n_en_shells):
        i = en_index * args.n_en_shells
        di = args.n_en_shells
        model_input = cdf_data.counts[:, :, :, i:i+di].astype(np.float32)
        latent[:, en_index, :] = models[i].encoder(model_input).numpy()        

    num_not_activating = (latent == 0).sum() / latent.size
    print(f'Fraction of zeros in latent representations: '
          f'{num_not_activating:.3f}')
    
    # Quantize latent representation. This section is very slow due to using
    # the bitstring array. If implemented in an FPGA it would be much faster
    # ------------------------------------------------------------------------
    latent[latent != 0] = roundbits(latent[latent != 0], args.mantissa_bits)
    
    latent_flat = latent.flatten()
    latent_bitarray = bitstring.BitArray()
    bar = progressbar.ProgressBar()
    
    for element in bar(latent_flat):
        # Special case handle zeros ------------------------------------------
        if element == 0:
            pad = 1 + 5 + args.mantissa_bits
            latent_bitarray.insert(
                bitstring.BitArray(uint=0, length=pad),
                len(latent_bitarray)
            )
            continue
        elif not np.isfinite(element):
            raise RuntimeError('Should not encounter NaN/Inf')

        # Continue ------------------------------------------------------------
        float_bitarray = bitstring.BitArray(bytes=element.tobytes()[::-1])
        
        sign_bitarray = float_bitarray[:1]                           # 1  it
        exp_bitarray = float_bitarray[1:6]                           # 5 bits
        mantissa_bitarray = float_bitarray[6:6+args.mantissa_bits]   # nbits

        # print(element, sign_bitarray.bin, exp_bitarray.bin,
        #       mantissa_bitarray.bin)
       
        latent_bitarray.insert(sign_bitarray, len(latent_bitarray))
        latent_bitarray.insert(exp_bitarray, len(latent_bitarray))        
        latent_bitarray.insert(mantissa_bitarray, len(latent_bitarray))

    # Apply entropy coding using the GZIP algorithm
    # ------------------------------------------------------------------------
    latent_gzipped = gzip.compress(latent_bitarray.bytes)
    improvement = len(latent_bitarray.bytes) / len(latent_gzipped)

    print(f'Entropy coding improved size by {improvement:.1f}X')

    # Write to disk
    # ------------------------------------------------------------------------    
    with open(args.out_file, 'wb') as fh:
        fh.write(header_bytes)
        fh.write(latent_gzipped)

    print(f'Wrote to {args.out_file}')

    # Report final compression ratio
    # ------------------------------------------------------------------------    
    cmpr_ratio = N_EN * N_PHI * N_THETA * 16 * cdf_data.epoch.size
    cmpr_ratio /= (len(header_bytes) + len(latent_gzipped)) * 8
    
    print(f'Final compression ratio: {cmpr_ratio:.1f}X')
    
        
def roundbits(fval, nbits):
    """Return the floating-point value `fval` rounded to `nbits` bits
    in the significand."""
    # https://stackoverflow.com/questions/54133181/round-a-x-float-to-y-with-a-reduced-mantissa-significant-floating-point
    significand, exponent = np.frexp(fval)
    scale = 2.0 ** nbits
    newsignificand = np.round_(significand * scale) / scale

    newexponent = exponent.copy()
    
    return np.ldexp(newsignificand, newexponent)

    
if __name__ == '__main__':
    main()
