## \file
## \ingroup tutorial_graphs
## \notebook -js
## \preview Create and draw a graph with error bars. If more graphs are needed, see the
## [gr03_err2gr.C](https://root.cern/doc/master/gerrors2_8C.html) tutorial
##
## See the [TGraphErrors documentation](https://root.cern/doc/master/classTGraphErrors.html)
##
## \macro_image
## \macro_code
## \author Rene Brun, Jamie Gooding

import numpy as np
import ROOT

c1 = ROOT.TCanvas("c1", "A Simple Graph with error bars", 200, 10, 700, 500)

c1.SetGrid()
c1.GetFrame().SetBorderSize(12)

#  We will use the constructor requiring: the number of points, arrays containing the x-and y-axis values, and arrays with the x- andy-axis errors

n = 10
x = np.array([-0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95])
y = np.array([1, 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1])
ex = np.array([0.05, 0.1, 0.07, 0.07, 0.04, 0.05, 0.06, 0.07, 0.08, 0.05])
ey = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8])

# If all x-axis errors should zero, just provide a single 0 in place of ex
gr = ROOT.TGraphErrors(n, x, y, ex, ey)

gr.SetTitle("TGraphErrors Example")
gr.SetMarkerColor(4)
gr.SetMarkerStyle(21)

# To draw in a new/empty canvas or pad, include the option "A" so that the axes are drawn (leave it out if the graph is to be drawn on top of an existing plot
gr.Draw("ALP")

c1.Update()
