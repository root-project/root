# Author: Enric Tejedor CERN  03/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import cppyy

def set_size(self, buf):
    # Parameters:
    # - self: graph object
    # - buf: buffer of doubles
    # Returns:
    # - buffer whose size has been set
    buf.reshape((self.GetN(),))
    return buf

# Create a composite pythonizor.
#
# A composite is a type of pythonizor, i.e. it is a callable that expects two
# parameters: a class proxy and a string with the name of that class.
# A composite is created with the following parameters:
# - A string to match the class/es to be pythonized
# - A string to match the method/s to be pythonized in the class/es
# - A callable that will post-process the return value of the matched method/s
#
# Here we create a composite that will match TGraph, TGraph2D and their error
# subclasses, and will pythonize their getter methods of the X,Y,Z coordinate
# and error arrays, which in C++ return a pointer to a double.
# The pythonization consists in setting the size of the array that the getter
# method returns, so that it is known in Python and the array is fully usable
# (its length can be obtained, it is iterable).
comp = cppyy.py.compose_method(
    '^TGraph(2D)?$|^TGraph.*Errors$',    # class to match
    'GetE?[XYZ](low|high|lowd|highd)?$', # method to match
    set_size)                            # post-process function

# Add the composite to the list of pythonizors
cppyy.py.add_pythonization(comp)

import ROOT
import numpy as np

def _create_graph(x=None, y=None, z=None, r=None, theta=None, 
                  xerr=None, yerr=None, xerr_asym=None, yerr_asym=None):
    """
    Create a ROOT TGraph object based on the provided parameters.

    Parameters:
    x, y (array-like): Data points for 2D graphs.
    z (array-like, optional): Data points for 3D graphs.
    r, theta (array-like, optional): Data points for polar graphs (radius and angle).
    xerr, yerr (array-like, optional): Symmetric errors for x and y axes.
    xerr_asym, yerr_asym (tuple of array-like, optional): Asymmetric errors for x and y axes.

    Returns:
    ROOT.TGraph or derived class: The created graph.
    """
    if r is not None and theta is not None:
        # Polar graph
        n_points = len(r)
        assert len(theta) == n_points, "r and theta must have the same length"
        graph = ROOT.TGraphPolar(n_points)
        for i in range(n_points):
            graph.SetPoint(i, theta[i], r[i])
        return graph

    if x is not None and y is not None:
        n_points = len(x)
        assert len(y) == n_points, "x and y must have the same length"

        if z is not None:
            # 3D graph
            assert len(z) == n_points, "x, y, and z must have the same length"
            graph = ROOT.TGraph2D(n_points)
            for i in range(n_points):
                graph.SetPoint(i, x[i], y[i], z[i])
            return graph

        if xerr_asym is not None or yerr_asym is not None:
            # Asymmetric errors
            xerr_low, xerr_high = xerr_asym if xerr_asym else (np.zeros(n_points), np.zeros(n_points))
            yerr_low, yerr_high = yerr_asym if yerr_asym else (np.zeros(n_points), np.zeros(n_points))
            graph = ROOT.TGraphAsymmErrors(n_points)
            for i in range(n_points):
                graph.SetPoint(i, x[i], y[i])
                graph.SetPointError(i, xerr_low[i], xerr_high[i], yerr_low[i], yerr_high[i])
            return graph

        if xerr is not None or yerr is not None:
            # Symmetric errors
            xerr = xerr if xerr is not None else np.zeros(n_points)
            yerr = yerr if yerr is not None else np.zeros(n_points)
            graph = ROOT.TGraphErrors(n_points)
            for i in range(n_points):
                graph.SetPoint(i, x[i], y[i])
                graph.SetPointError(i, xerr[i], yerr[i])
            return graph

        # Simple graph
        graph = ROOT.TGraph(n_points)
        for i in range(n_points):
            graph.SetPoint(i, x[i], y[i])
        return graph

    raise ValueError("Invalid combination of parameters. Provide appropriate inputs for a graph.")
