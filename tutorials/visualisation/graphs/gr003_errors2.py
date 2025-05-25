## \file
## \ingroup tutorial_graphs
## \notebook -js
## \preview Create and draw two graphs with error bars, superposed on the same canvas
##
## We first draw an empty frame with the axes, then draw the graphs on top of it
## Note that the graphs should have the same or very close ranges (in both axis),
## otherwise they may not be visible in the frame.
##
## Alternatively, an automatic axis scaling can be achieved via a
## [TMultiGraph](https://root.cern/doc/master/classTMultiGraph.html)
##
## See the [TGraphErrors documentation](https://root.cern/doc/master/classTGraphErrors.html)
##
## \macro_image
## \macro_code
## \author Rene Brun, Jamie Gooding

import numpy as np
import ROOT

c1 = ROOT.TCanvas("c1", "2 graphs with errors", 200, 10, 700, 500)
c1.SetGrid()

# draw a frame to define the range
hr = c1.DrawFrame(-0.4, 0, 1.2, 12)
hr.SetXTitle("X title")
hr.SetYTitle("Y title")
c1.GetFrame().SetBorderSize(12)

# create first graph
# We will use the constructor requiring: the number of points, arrays containing the x-and y-axis values, and arrays with the x- andy-axis errors
n1 = 10
xval1 = np.array([-0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95])
yval1 = np.array([1, 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1])
ex1 = np.array([0.05, 0.1, 0.07, 0.07, 0.04, 0.05, 0.06, 0.07, 0.08, 0.05])
ey1 = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8])
# If all x-axis errors should zero, just provide a single 0 in place of ex1
gr1 = ROOT.TGraphErrors(n1, xval1, yval1, ex1, ey1)
gr1.SetMarkerColor(ROOT.kBlue)
gr1.SetMarkerStyle(21)
# Since we already have a frame in the canvas, we draw the graph without the option "A" (which draws axes for this graph)
gr1.Draw("LP")

# create second graph
n2 = 10
xval2 = np.array([-0.28, 0.005, 0.19, 0.29, 0.45, 0.56, 0.65, 0.80, 0.90, 1.01])
yval2 = np.array([0.82, 3.86, 7, 9, 10, 10.55, 9.64, 7.26, 5.42, 2])
ex2 = np.array([0.04, 0.12, 0.08, 0.06, 0.05, 0.04, 0.07, 0.06, 0.08, 0.04])
ey2 = np.array([0.6, 0.8, 0.7, 0.4, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7])
gr2 = ROOT.TGraphErrors(n2, xval2, yval2, ex2, ey2)
gr2.SetMarkerColor(ROOT.kRed)
gr2.SetMarkerStyle(20)
gr2.Draw("LP")
