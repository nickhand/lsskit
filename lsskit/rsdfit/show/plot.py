from pyRSD.rsdfit.util import plot

__all__ = ['compare_theory_fits']

def compare_theory_fits(ax, x, y, labels=["", ""]):
    """
    Compare two RSD fits, overlaying the two theory
    results over the data
    
    Parameters
    ----------
    x : FittingResult
        the first theory model to plot
    y : FittingResult
        the second theory model to plot
    labels : list, optional
        list of the labels to apply to `x` and `y`
    
    Returns 
    -------
    ax : Axes
        the axes instance
    """
    colors = ax.color_list[:x.driver.data.size]
        
    if len(labels) != 2:
        raise ValueError("please provide exactly two labels for the comparison plot")

    # determine the offset
    offset = -0.1 if x.driver.mode == 'pkmu' else 0.

    # plot x first and save the normalization
    x.driver.set_fit_results()
    if x.driver.mode == 'pkmu':
        norm = plot.pkmu_normalization(x.driver)
    else:
        norm = plot.poles_normalization(x.driver)
    plot.plot_normalized_theory(ax, x.driver, offset=offset, color=colors, ls='--', label=labels[0])
    
    # plot y second
    y.driver.set_fit_results()
    plot.plot_normalized_theory(ax, y.driver, offset=offset, color=colors, norm=norm, label=labels[1])

    # and the data
    plot.plot_normalized_data(ax, x.driver, offset=offset, c='0.7', use_labels=False, alpha=0.4, zorder=1, norm=norm)

    # format and show
    plot.add_axis_labels(ax, x.driver)
    
    # add a legend
    ax.legend(loc=0)
    return ax