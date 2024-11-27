# Author: Aaron Jomy CERN  11/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

def get_hist_constructor(dims):
    """
    return a callable ROOT.TH* constructor based on the data dimensionality 
    """

    import ROOT

    histogram_constructor_map = {
    '1': ROOT.TH1D,
    '2': ROOT.TH2D,
    '3': ROOT.TH3D
    }

    return histogram_constructor_map.get(dims)

def _CreateHistogram(name, title, bins, nrange=None, data=None, density=False, weights=None):
    """
    return an initialised TH* object filled with data
    """

    import numpy as np
    
    try:
        N, D = data.shape
    except (AttributeError, ValueError):
        data = np.atleast_2d(data).T
        N, D = data.shape

    if weights is not None:
        weights = np.asarray(weights)
        assert weights.shape == data.shape, "Weights should match the shape of the data"

    assert D in {1, 2, 3}, "Only 1D, 2D, and 3D histograms are supported"

    is_variable_bins = [False] * D
    bins = list(bins)
    if bins is None:
        bins = [100] * D  # default to 100 bins

    if isinstance(bins, int):
        bins = [bins] * D  # apply same bin count to all dimensions

    if isinstance(bins, (list, np.ndarray)):
        for i in range(D):
            if (isinstance(bins[i], (list, np.ndarray))):
                is_variable_bins[i] = True

    assert len(bins) == D, f"Bins tuple must have {D} elements"

    # Handle range for each dimension
    if nrange is None:
        if D == 1:
            nrange = [[data.min(), data.max()]]
        else:
            nrange = [[data[:, i].min(), data[:, i].max()] for i in range(D)]

    else:
        assert len(nrange) == D, f"range must have {D} elements"
        for i in range(D):
            if nrange[i] is None:
                nrange[i] = [data[:, i].min(), data[:, i].max()]
            assert len(nrange[i]) == 2, f"range for dim {i} must have exactly two elements"
            assert nrange[i][0] <= nrange[i][1], f"range lower bound for dim {i} must be <= upper bound"

    if weights is None:
        weights = np.ones(data.shape[0])
    
    args = []
    print(is_variable_bins)
    for i in range(D):
        if is_variable_bins[i]:
            edges = np.asarray(bins[i]) 
            args.extend([len(edges) - 1, edges])
        else:
            args.extend([int(bins[i]), nrange[i][0], nrange[i][1]])
    
    root_class = get_hist_constructor(str(D))
    if root_class is None:
        raise ValueError(f"No ROOT histogram class for: {D}D data")

    hist = root_class(name, title, *args)
    
    # ideally we should use FillN, but we can remove this loop once Fill is pythonised to handle both scalars and vectors)
    if (data):
        for i, point in enumerate(data):
            hist.Fill(*point, weights[i]) 

    return hist