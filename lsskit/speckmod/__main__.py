"""
    tools.py
    lsskit.speckmod
    
    __author__ : Nick Hand
    __desc__ : tools for helping with modeling of power spectra
"""
import fitit
from .. import numpy as np
from . import plugins
import argparse 

def perform_fit():
    
    # parse the input arguments
    parser = fitit.arg_parser.initialize_parser()
    parser.formatter_class = argparse.RawTextHelpFormatter

    # the input data  
    h = "the input data, specified as:\n\n"
    parser.add_argument('--data', required=True, type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('data'))
    
    # the input model  
    # h = "the input model, specified as:\n\n"
    # parser.add_argument('--model', required=True, type=plugins.ModelInput.parse,
    #                        help=h+plugins.ModelInput.format_help('model'))
    
    # parse
    args = fitit.arg_parser.parse_command_line(parser.parse_args())
    if args['subparser_name'] not in ['chisq', 'mcmc']:
        raise NotImplementedError("Only `chisq` and `mcmc` subparsers allowed; not %s" %args['subparser_name'])
    args['no_file_logger'] = True
    
    for i, j, k in args['data']:
        print i, j, k
    
    #
    # # get the parameters
    # params, logfile = fitit.run.initialize_params(args)

    
    # # get the theory and model
    # model, theory_params = fitit.get_theory(params['theory'])
    #
    # # a few variables we need
    # kmin = params['driver']['kmin'].value
    # kmax = params['driver']['kmax'].value
    # mode = params['driver']['mode'].value
    # lam_type = params['driver']['type'].value
    #
    # # loop over each set of simulations to fit
    # sim_name = params['driver']['sim'].value
    # print "Fitting %s simulations..." %sim_name
    #
    # # get the sim data
    # data = get_sim_data(sim_name)
    #
    # # get P_mm data
    # d_mm = data['mm']
    #
    # # do all mass bins
    # bin_num = params['driver']['bin_num'].value
    # if bin_num is None:
    #     bin_nums = range(len(data))
    # else:
    #     bin_nums = [bin_num]
    # for bin_num in bin_nums:
    #
    #     # halo-matter and halo-halo
    #     d_hm = data['hm%d' %bin_num]
    #     d_hh = data['hh%d' %bin_num]
    #
    #     # get the lambda, enforcing the k_min and k_max
    #     data_df = get_lambda(lam_type, d_mm, d_hm, d_hh, d_hm.bias, k_min=kmin, k_max=kmax)
    #
    #     print "Mass Bin #%d, b1 = %.2f" %(bin_num, d_hm.bias)
    #     print "___________________"
    #
    #     # do the fit
    #     kwargs = {}
    #     if mode == 'chisq':
    #         result = fitit.driver.chisq_run(params['driver'], data_df, theory_params, model.copy())
    #     elif mode == 'mcmc':
    #         result = fitit.driver.mcmc_run(params['driver'], data_df, theory_params, model.copy(), pool=None)
    #         kwargs['walkers'] = result.walkers
    #         kwargs['iterations'] = result.iterations
    #
    #     if params['driver']['silent'].value:
    #         result.summarize_fit(to_screen=True)
    #
    #     # save
    #     if 'chain_number' in params['driver']:
    #         params['driver']['chain_number'].value =  "bin%d" %(bin_num)
    #     else:
    #         params['driver']['chain_number'] = fitit.lmfit.Parameter(value="bin%d" %(bin_num))
    #     output_name = fitit.iotools.create_output_file(params['driver'], mode, **kwargs)
    #     fitit.iotools.save_pickle(result, output_name)
    #
    # # now let's save the params too
    # folder = params['driver']['folder'].value
    # fitit.iotools.save_pickle(params, os.path.join(folder, 'params.pickle'))

