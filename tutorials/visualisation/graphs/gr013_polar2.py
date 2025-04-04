## \file
## \ingroup tutorial_graphs
## \notebook
## \preview Create and draw a polar graph with errors and polar axis in radians (PI fractions).
## See the [TGraphPolar documentation](https:#root.cern/doc/master/classTGraphPolar.html)
##
## Since TGraphPolar is a TGraphErrors, it is painted with
## [TGraphPainter](https:#root.cern/doc/master/classTGraphPainter.html) options.
##
## With GetPolargram we retrieve the polar axis to format it see the
## [TGraphPolargram documentation](https:#root.cern/doc/master/classTGraphPolargram.html)
##
## \macro_image
## \macro_code
## \author Olivier Couet, Jamie Gooding

import numpy as np
import ROOT

CPol = ROOT.TCanvas("CPol", "TGraphPolar Example", 500, 500)

theta = np.array([])
radius = np.array([])
etheta = np.array([])
eradius = np.array([])

for i in range(8):
    theta = np.append(theta, (i + 1) * (np.pi / 4.0))
    radius = np.append(radius, (i + 1) * 0.05)
    etheta = np.append(etheta, np.pi / 8.0)
    eradius = np.append(eradius, 0.05)

grP1 = ROOT.TGraphPolar(8, theta, radius, etheta, eradius)
grP1.SetTitle("")

grP1.SetMarkerStyle(20)
grP1.SetMarkerSize(2.0)
grP1.SetMarkerColor(4)
grP1.SetLineColor(2)
grP1.SetLineWidth(3)
# Draw with polymarker and errors
grP1.Draw("PE")

# To format the polar axis, we retrieve the TGraphPolargram.
# First update the canvas, otherwise GetPolargram returns 0
CPol.Update()
if grP1.GetPolargram():
    grP1.GetPolargram().SetToRadian()  # tell ROOT that the theta values are in radians
