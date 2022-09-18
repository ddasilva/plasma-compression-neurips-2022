import functools

import h5py
import numpy as np

def flatten_skymap(frame):
    n_az, n_el, n_en = 32, 16, 32
    img = np.zeros((n_az, n_en * n_el))
    
    for i in range(n_en):
        img[:, i*n_el:(i+1)*n_el] = frame[:, :, i] 
    
    return img


def get_f1ct(test_data, phases):
    # comment: because counts are inferred from distribution and
    # not in the body frame, each f1ct represents an average over
    # all connected deflection states for that sample.
    f1ct = np.zeros_like(test_data[phases[0]]['dist'][0])
    inc = np.zeros_like(f1ct)

    for phase in phases:
        for i in range(test_data[phase]['dist'].shape[0]):
            dist = test_data[phase]['dist'][i]
            counts = test_data[phase]['counts'][i]

            mask = counts>0
            f1ct[mask] += dist[mask] / counts[mask]
            inc[mask] += 1
        
    f1ct[inc>0] /= inc[inc>0]
    f1ct[inc==0] = 0
    
    return f1ct


@functools.lru_cache(maxsize=None)
def load_test_data(phase, filename=None):
    """Load test data

    Returns
      dictionary of test data
    """
    if filename is None:
        filename = '/mnt/efs/dasilva/compression-cfha/data/samples_test_n=5000.hdf'

    test_data = {}

    hdf = h5py.File(filename, 'r')

    if phase == 'all':
        phases = list(hdf.keys())
    else:
        phases = [phase]

    test_data['counts'] = np.concatenate([hdf[phase]['counts'][:] for phase in phases])
    test_data['dist'] = np.concatenate([hdf[phase]['dist'][:] for phase in phases])
    test_data['phi'] = np.concatenate([hdf[phase]['phi'][:] for phase in phases])
    test_data['theta'] = np.concatenate([hdf[phase]['theta'][:] for phase in phases])
    test_data['E'] = np.concatenate([hdf[phase]['E'][:] for phase in phases])

    if len(phases) > 1:
        n = hdf[phases[0]]['theta'].shape[0]

        I = np.arange(n)
        np.random.shuffle(I)
        I = I[:n]

        for key in test_data.keys():
            test_data[key] = test_data[key][I]

    hdf.close()
            
    return test_data
