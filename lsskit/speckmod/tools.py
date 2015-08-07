"""
    tools.py
    lsskit.speckmod
    
    __author__ : Nick Hand
    __desc__ : tools for helping with modeling of power spectra
"""
from .. import numpy as np
 
#------------------------------------------------------------------------------       
def convert_arg_line_to_args(self, line):
    """
    Custom function that reads arguments from file, to be used
    as the ``argparse.ArgumentParser.convert_arg_line_to_args`` 
    function
    """
    if line[0] == '#': return
    r = line.find(' #')
    if r >= 0:
        line = line[:r] 
    r = line.find('\t#')
    if r >= 0:
        line = line[:r] 

    line = line.strip()
    if len(line) == 0: return
    yield line

#------------------------------------------------------------------------------
def get_valid_data(data, kmin=None, kmax=None):
    """
    Return the valid data. First, any NaN entries are removed
    and if ``kmin`` or ``kmax`` are not ``None``, then the 
    ``k`` column in ``data`` is used to trim the valid range.
    
    Parameters
    ----------
    data : PkmuResult or PkResult
        The power data instance holding the `k`, `power`, and
        optionally, the `error` data columns
    kmin : float, optional
        minimum wavenumber to trim by (inclusively), in h/Mpc
    kmax : float, optional
        maximum wavenumber to trim by (inclusively), in h/Mpc
    
    Returns
    -------
    toret : dict
        dictionary holding the trimmed data arrays, with keys
        ``k``, ``power``, and optionally, ``error``.
    """
    columns = ['k', 'power', 'error']
    
    valid = ~np.isnan(data['power'])
    if kmin is not None:
        valid &= (data['k'] >= kmin)
    if kmax is not None:
        valid &= (data['k'] <= kmax)
    
    toret = {}
    for col in columns:
        if col in data:
            toret[col] = data[col][valid]

    return toret

#------------------------------------------------------------------------------
def make_param_table(param_names, dims, coords):
    """
    Return an empty ``pandas.DataFrame``, indexed by a ``MultiIndex``
    specified by ``dims`` and ``coords`` and with column names for 
    each parameter (and error) in ``param_names``.
    
    Parameters
    ----------
    param_names : list of str
        the names of the parameters, which will serve as the column names
    dims : list of str
        the names of the dimensions of the MultiIndex
    coordinates : list, xray.core.coordinates.DataArrayCoordinates
        list of coordinates for each dimension. `itertools.product` of
        each coordinate axis is used to make the MultiIndex
    
    Returns
    -------
    df : pandas.DataFrame
        an empty DataFrame to store the value and error on each parameter
    """
    import itertools
    import pandas as pd
    
    if hasattr(coords, 'values'):
        coords = [x.values for x in coords.values()]
        
    param_plus_errs = list(itertools.chain(*[(p, p+"_err") for p in param_names]))
    index = list(itertools.product(*[coords[i] for i in range(len(dims))]))
    index = pd.MultiIndex.from_tuples(index, names=dims)
    return pd.DataFrame(index=index, columns=param_plus_errs)

#------------------------------------------------------------------------------    
def compare_bestfits(mode, **kwargs):
    """
    Compare the best-fit parameters to the data.
    
    Parameters
    ----------
    mode : str, {`function`, `params`, `gp`, `spline`}
        either compare to bestfit function or bestfit params
    kwargs : key/value pairs
        data : subclass of lsskit.speckmod.plugins.ModelInput
            plugin instance specifying the data
        model : subclass of lsskit.speckmod.plugins.ModelInput
            plugin instance specifying the model, only needed
            for `mode == params`
        bestfit_file : str
            the name of the file holding the pickled dataframe
        select : list of str
            a list holding strings with the format should 
            `index_col`:value'. This specifies which bin to be
            compared
    """
    import plotify as pfy
    import pandas as pd
    if mode == 'gp':
        from pyRSD.rsd.mu0_modeling import GPModelParams
    elif mode == 'spline':
        from pyRSD.rsd.mu0_modeling import SplineTableModelParams
        
    if mode not in ['function', 'params', 'gp', 'spline']:
        raise ValueError("``mode`` in compare_bestfits must be `function`, `params`, `gp`, or `spline`")
    
    # make the index cols
    try:
        index_cols = [x.split(':')[0] for x in kwargs['select']]
        select = [int(x.split(':')[1]) for x in kwargs['select']]
    except:
        raise ValueError("``select`` should have format: `index_col`:value")
    
    # read the bestfits file and select 
    df = pd.read_pickle(kwargs['bestfit_file'])
    valid = index_cols
    if mode == 'function': valid += ['k']
    if not all(x in df.columns for x in valid):
        raise ValueError("please specify a bestfit file with columns: %s" %(", ".join(valid)))
    df = df.set_index(valid)
    
    # get the key dictionary and print out what we are selecting
    key = dict((df.index.names[i], df.index.levels[i][v]) for i, v in enumerate(select))
    msg = ", ".join("%s = %s" %(k,v) for k,v in key.iteritems())
    print "selecting " + msg
    
    # select the bestfit
    select = tuple(df.index.levels[i][v] for i, v in enumerate(select))
    
    # load the GP
    if mode == 'gp':
        
        gp = GPModelParams(kwargs['gp_file'])
        args = tuple(df.loc[select, col] for col in kwargs['interp_cols'])
        bestfits = gp.to_dict(*args)
        print "gp bestfit values:\n-------------"
        print "\n".join("%s = %s" %(k,str(v)) for k,v in bestfits.iteritems())
        
        actual = {k:df.loc[select, k] for k in kwargs['model'].param_names}
        print "actual bestfit values:\n-------------"
        print "\n".join("%s = %s" %(k,str(v)) for k,v in actual.iteritems())
    
    # load the spline table
    elif mode == 'spline':    
        
        table = SplineTableModelParams(kwargs['spline_file'])
        s8_z = df.loc[select, 's8_z']
        b1 = df.loc[select, 'b1']
        bestfits = table(s8_z, b1)
        print "spline table bestfit values:\n-------------"
        print "\n".join("%s = %s" %(k,str(v)) for k,v in bestfits.iteritems())
        
        actual = {k:df.loc[select, k] for k in kwargs['model'].param_names}
        print "actual bestfit values:\n-------------"
        print "\n".join("%s = %s" %(k,str(v)) for k,v in actual.iteritems())
        
    elif mode == 'params':
        bestfits = {k:df.loc[select, k] for k in kwargs['model'].param_names}
        
    if mode == 'params' or mode == 'gp' or mode == 'spline':
        kwargs['data'].select = key
        
        # this should hopefully only loop over one thing
        for key, extra, data_df in kwargs['data']:    
        
            # plot the data
            pfy.errorbar(data_df.index.values, data_df['y'], data_df['error'])
    
            # plot the bestfit parameters
            x = data_df.index.values
            y = kwargs['model'](x, **dict(bestfits, **extra))
            lines = pfy.plot(x, y)
            
    else: # mode is `function`
        df = df.xs(select)

        # select the data
        data_df = kwargs['data'].to_dataframe(key)
    
        # plot the data
        pfy.errorbar(data_df.index.values, data_df['y'], data_df['error'])
    
        # plot the bestfit function mean
        x = df.index.values
        y = df['mean']
        errs = df['error']
        lines = pfy.plot(x, y)
        pfy.plt.fill(np.concatenate([x, x[::-1]]),
                     np.concatenate([y - errs,
                                    (y + errs)[::-1]]),
                                    alpha=.5, fc=lines[0].get_color(), ec='None')

                                
    ax = pfy.gca()
    ax.title.update('Bestfit (%s) comparison for %s' %(mode,msg), fontsize=16)
    ax.xlabel.update(r"$k$ ($h$/Mpc)", fontsize=16)
    ax.ylabel.update(kwargs['data'].variable_str, fontsize=16)
    pfy.show()
    
#------------------------------------------------------------------------------
def add_bestfit_param(**kwargs):
    """
    Add a bestfit parameter to the specified dataframe
    
    Parameters
    ----------
    kwargs : key/value pairs
        bestfit_file : str
            the name of the file holding the pickled dataframe
        expr : str
            the expression for the new parameter; should be of 
            the form ``param_name = expr``
        output : str, optional
            the name for the new dataframe; if none provided, 
            the input dataframe is overwritten
        error : str, {'fractional', 'absolute'}
            the type of errors to use
    """
    import pandas as pd
    
    # load the parameters
    df = pd.read_pickle(kwargs['bestfit_file'])
    
    # parse the expression
    param_name, expr = kwargs['expr'].split("=")
    param_name, expr = param_name.strip(), expr.strip()
    
    # find param names 
    depends = []
    for col in df.columns:
        if '_err' in col:
            continue
        if col in expr:
            depends.append(col)
        
    # get the eval'ed expression
    eval_expr = expr
    for d in depends:
        eval_expr = eval_expr.replace(d, "df['%s']" %d)
    df[param_name] = eval(eval_expr)
    
    # now compute the error
    if kwargs['error'] == 'fractional':
        var = 0
        for d in depends:
            var += (df[d+'_err']/df[d])**2
        df[param_name+"_err"] = df[param_name] * var**0.5
    else:
        var = 0
        for d in depends:
            var += df[d+'_err']**2
        df[param_name+"_err"] = var**0.5
    
    # save the output
    if kwargs.get('output', None) is None:
        df.to_pickle(kwargs['bestfit_file'])
    else:
        df.to_pickle(kwargs['output'])