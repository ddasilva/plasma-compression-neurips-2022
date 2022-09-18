import h5py
import numpy as np

# Load weights
WEIGHTS = {}

hdf = h5py.File('moms_weights.hdf', 'r')
for key in hdf.keys():
    WEIGHTS[key] = hdf[key][:]
hdf.close()

# Constants
mass = 1.67262178e-24
kb = 1.3807e-16
temp1eV =  1.1604e4
energy1eV = 1.60217657e-12


def fast_moments(psd):
    """Moments integration code converted from the MMS/FPI ground system.
    
    Assumes the standard MMS grid.

    Args
       psd: phase space density
    Returns
       dictionary containing keys n, vx, vy, vz, txx, tyy, tzz, txy, txz, tyz.
    """
    results = {key: np.sum(WEIGHTS[key] * psd) for key in WEIGHTS.keys()}

    results['vx'] = results['nvx'] / 1e5 / results['n']
    results['vy'] = results['nvy'] / 1e5 / results['n']
    results['vz'] = results['nvz'] / 1e5 / results['n']    
    results['txx'] = mass*results['ntxx']/results['n']/kb/temp1eV - 1e10*mass*results['vx']**2/kb/temp1eV
    results['tyy'] = mass*results['ntyy']/results['n']/kb/temp1eV - 1e10*mass*results['vy']**2/kb/temp1eV
    results['tzz'] = mass*results['ntzz']/results['n']/kb/temp1eV - 1e10*mass*results['vz']**2/kb/temp1eV
    results['txy'] = mass*results['ntxy']/results['n']/kb/temp1eV - 1e10*mass*results['vx']*results['vy']/kb/temp1eV
    results['txz'] = mass*results['ntxz']/results['n']/kb/temp1eV - 1e10*mass*results['vx']*results['vz']/kb/temp1eV
    results['tyz'] = mass*results['ntyz']/results['n']/kb/temp1eV - 1e10*mass*results['vy']*results['vz']/kb/temp1eV
    
    return results
