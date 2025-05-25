## \file
## \ingroup tutorial_graphs
## \notebook
## \preview This tutorial demonstrates the use of TGraphAsymmErrors to plot a graph with asymmetrical errors on both the x and y axes.
## The errors for the x values are divided into low (left side of the marker) and high (right side of the marker) errors.
## Similarly, for the y values, there are low (lower side of the marker) and high (upper side of the marker) errors.
##
## \macro_image
## \macro_code
##
## \author Miro Helbich, Jamie Gooding

import numpy as np
import ROOT

c2 = ROOT.TCanvas("c2", "", 700, 500)

c2.SetGrid()
npoints = 3
xaxis = np.array([1.0, 2.0, 3.0])
yaxis = np.array([10.0, 20.0, 30.0])

exl = np.array([0.5, 0.2, 0.1])  # Lower x errors
exh = np.array([0.5, 0.3, 0.4])  # Higher x errors
eyl = np.array([3.0, 5.0, 4.0])  # Lower y errors
eyh = np.array([3.0, 5.0, 4.0])  # Higher y errors

gr = ROOT.TGraphAsymmErrors(
    npoints, xaxis, yaxis, exl, exh, eyl, eyh
)  # Create the TGraphAsymmErrors object with data and asymmetrical errors

gr.SetTitle("A simple graph with asymmetrical errors")
gr.Draw("A*")  # "A" = draw axes and "*" = draw markers at the points with error bars
