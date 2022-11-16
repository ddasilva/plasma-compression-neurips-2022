
import argparse
import gzip
import os

import bitstring
import h5py
import numpy as np
import progressbar

from nnet_train import (
    N_EN, N_PHI, N_THETA,
    load_model,
)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    # Parse command line arguments
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('compressed_file')
    parser.add_argument('out_file')
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-en-shells', type=int, required=True)
    parser.add_argument('--hidden-layer-size', type=int, required=True)
    parser.add_argument('--mantissa-bits', type=int, default=4)

    args = parser.parse_args()

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

    # Load binary data from disk and remove entropy coding
    # ------------------------------------------------------------------------
    fh = open(args.compressed_file, 'rb')
    ntimes_bytes = fh.read(4)                   # header
    ntimes = np.frombuffer(ntimes_bytes, dtype='>u4')[0]

    means_bytes = ntimes * N_EN * 2  # float16
    means_bytes = fh.read(means_bytes)
    means = np.frombuffer(means_bytes, dtype=np.float16)
    means = means.reshape((ntimes, N_EN))
    
    latent_bytes = gzip.decompress(fh.read())

    print('Entropy coding removed')
    print('ntimes = ', ntimes)
    
    # Convert quantized latent representaion floats to float16, and stored in
    # flattened latent array
    # ------------------------------------------------------------------------
    # Convert the bytes read from the file to a BitArray for convinience
    latent_bitarray = bitstring.BitArray(bytes=latent_bytes)

    element_nbits = 1 + 5 + args.mantissa_bits
    assert len(latent_bitarray) % element_nbits == 0
    
    n_float16s = len(latent_bitarray) // element_nbits
    bar = progressbar.ProgressBar()
    latent_flat = []

    # Loop thorugh the elements of the latent bitarray and convert each
    # one to a float16
    for i in bar(np.arange(n_float16s)):        
        start = i * element_nbits
        stop = (i + 1) * element_nbits
        element_bitarray = latent_bitarray[start:stop]

        if element_bitarray.uint == 0:
            # Special case 0 because it happens so much. Slight speedup.
            latent_flat.append(0)
            continue
        
        sign_bitarray = element_bitarray[:1]
        exp_bitarray = element_bitarray[1:6]
        mantissa_bitarray = element_bitarray[6:]

        # pad mantissa for float16 length (10 bits)
        mantissa_pad = 10 - args.mantissa_bits
        if mantissa_pad > 0:
            mantissa_bitarray.insert(
                bitstring.BitArray(uint=0, length=mantissa_pad),
                len(mantissa_bitarray)
            )
        assert len(mantissa_bitarray) == 10
        
        float16_bitarray = bitstring.BitArray()
        float16_bitarray.insert(sign_bitarray, len(float16_bitarray))
        float16_bitarray.insert(exp_bitarray, len(float16_bitarray))
        float16_bitarray.insert(mantissa_bitarray, len(float16_bitarray))

        element = np.frombuffer(float16_bitarray.bytes[::-1],
                                dtype=np.float16)[0]
        latent_flat.append(element)

    latent_flat = np.array(latent_flat, dtype=np.float32)

    # Convert flat latent representation to shaped array
    # ------------------------------------------------------------------------
    latent_shape = (
        ntimes, N_EN // args.n_en_shells, args.hidden_layer_size
    )
    
    latent = latent_flat.reshape(latent_shape)
    
    # Use model to decode the latent representation
    # --------------------------------------------------
    counts = np.zeros((ntimes, N_PHI, N_THETA, N_EN))
    
    for en_index in range(N_EN // args.n_en_shells):
        i = en_index * args.n_en_shells
        di = args.n_en_shells

        model_input = latent[:, en_index, :]
        model_output = models[i].decoder(model_input).numpy()

        counts[:, :, :, i:i+di] = model_output 

    print('Latent represention decoded with NNet')

    # Normalize to means
    # -----------------------------------------------------------------------
    for en_index in range(N_EN):
        for j in range(ntimes):
            mean_orig = means[j, en_index]
            mean_recon = counts[j, :, :, en_index].mean()

            if mean_orig == 0:
                counts[j, :, :, en_index] = 0
            elif mean_recon > 0:                
                counts[j, :, :, en_index] *= mean_orig / mean_recon
    
    # Write counts to HDF5 file
    # ------------------------------------------------------------------------
    hdf = h5py.File(args.out_file, 'w')
    hdf['counts'] = counts
    hdf.close()

    print(f'Wrote decompressed counts to output file {args.out_file}')

    
if __name__ == '__main__':
    main()
