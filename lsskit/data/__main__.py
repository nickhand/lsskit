"""
    __main__.py
    lsskit.data

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : data-related functions to install as console scripts
"""
from .. import numpy as np
from . import tools as data_tools
from ..specksis import tools
import argparse

def save_runPB_galaxy_stats():
    """
    Compute the statistics of the galaxy mock catalogs for the
    HOD runPB simulations at z = 0.55, and save the resulting
    dictionary as a pickle file
    """
    import pickle
    from lsskit.catio import HODMock
    import collections
    
    # parse the input arguments
    desc = "compute galaxy statistics for each runPB mock and save"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('pattern', type=str, help='the pattern to load the runPB catalog')
    parser.add_argument('output_file', type=str, help="the name of the output file")
    args = parser.parse_args()
    
    N_mocks = 10
    mock_base = args.pattern
    V = 1380.**3
    info = collections.defaultdict(list)
    for mock_num in range(N_mocks):
        
        print "processing mock #%d..." %mock_num
        mock = HODMock.from_hdf(mock_base %mock_num)

        # get the total number
        mock.clear_restrictions()

        i = {}
        N_tot = float(mock.total_galaxies)
        N_c, N_cA, N_cB = map(float, mock.centrals_totals())
        N_s, N_sA, N_sB = map(float, mock.satellites_totals())

        # now store the results
        info['N_tot'].append(N_tot)
        info['N_c'].append(N_c)
        info['N_s'].append(N_s)
        info['N_cB'].append(N_cB)
        info['N_sB'].append(N_sB)
        
        # 1-halo pairs
        info['cen_sat_pairs'].append(mock.cen_sat_pairs)
        info['sat_sat_pairs'].append(mock.sat_sat_pairs)


    def compute_stats(**kw):
        i = {}
        i['N_tot'] = kw['N_tot']
        
        # fractions
        i['fs'] = kw['N_s']/kw['N_tot']
        i['fcB'] = kw['N_cB']/kw['N_c']
        i['fsB'] = kw['N_sB']/kw['N_s']
        
        # 1-halo amplitudes
        nbar = kw['N_tot'] / kw['V']
        i['NcBs'] = (kw['cen_sat_pairs'] / kw['V'] / nbar**2) / (2*i['fs']*(1-i['fs'])*i['fcB'])
        i['NsBsB'] = (kw['sat_sat_pairs'] / kw['V'] / nbar**2) / (i['fs']*i['fsB'])**2
        
        return i
    
    # save the output for each 10
    toret = {}
    for i in range(N_mocks):
        kwargs = {k : info[k][i] for k in info.keys()}
        toret[i] = compute_stats(V=V, **kwargs)
        
    # and do the mean
    allsum = {}
    for k in info.keys():
        allsum[k] = np.sum(info[k])
    toret['mean'] = compute_stats(V=N_mocks*V, **allsum)
            
    # and save
    pickle.dump(toret, open(args.output_file, 'w'))
    
def compute_biases():
    """
    Compute the linear biases from set of cross/auto realspace spectra
    """
    from lsskit.data import tools
    import pickle
                    
    # parse the input arguments
    desc = "compute the linear biases from set of cross/auto realspace spectra"
    parser = argparse.ArgumentParser(description=desc)
    
    # required arguments
    h = data_tools.PowerSpectraParser.format_help()
    parser.add_argument('data', type=data_tools.PowerSpectraParser.data, help=h)
    h = data_tools.PowerSpectraCallable.format_help()
    parser.add_argument('Pxm_callable', type=data_tools.PowerSpectraCallable.data, help=h)
    h = data_tools.PowerSpectraCallable.format_help()
    parser.add_argument('Pmm_callable', type=data_tools.PowerSpectraCallable.data, help=h)
    h = 'the name of the output file'
    parser.add_argument('-o', '--output', type=str, required=True, help=h)
    
    # options
    h = "only consider a subset of keys; specify as ``-s a = '0.6452', '0.7143'``"
    parser.add_argument('-s', '--subset', type=str, action=data_tools.StoreDataKeys, default={}, help=h)
    h = "aliases to use for the keys; specify as ``--aliases sample = cc:cen, gg:gal"
    parser.add_argument('--aliases', type=str, action=data_tools.AliasAction, default={}, help=h)
    args = parser.parse_args()
    
    # the spectra
    Pxm = getattr(args.data, args.Pxm_callable['name'])(**args.Pxm_callable['kwargs'])
    Pmm = getattr(args.data, args.Pmm_callable['name'])(**args.Pmm_callable['kwargs'])
    
    def squeeze(tup):
        if not (len(tup)-1):
            return tup[0]
        else:
            return tup
    
    def determine_bias(k, data):
        inds = (k >= 0.01)&(k <= 0.04)
        return np.mean(data[inds])
    
    # loop over each
    toret = {}
    for key in Pxm.ndindex():
        valid = True
        for k in args.subset:
            if key[k] not in args.subset[k]:
                valid = False
                break
        if not valid:
            continue
            
        tup = squeeze(list(key[k] for k in Pxm.dims))
        subkey = {k:key[k] for k in Pxm.dims if k in Pmm.dims}
        
        x = Pxm.sel(**key).values
        y = Pmm.sel(**subkey).values
        y_shot = tools.get_Pshot(y)
        
        ratio = x['power']/(y['power'] - y_shot)
        b1 = determine_bias(x['k'], ratio)
        
        # substitute any alias
        for i, v in enumerate(tup):
            dim = Pxm.dims[i]
            if dim in args.aliases:
                if v in args.aliases[dim]:
                    tup[i] = args.aliases[dim][v]
        toret[tuple(tup)] = b1
        
    pickle.dump(toret, open(args.output, 'w'))
        
        
    
        
