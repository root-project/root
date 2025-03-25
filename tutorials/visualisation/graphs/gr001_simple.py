## \file
## \ingroup tutorial_graphs
## \notebook
## \preview This tutorial demonstrates how to create simple graphs in ROOT. The example is divided into two sections:
## - The first section plots data generated from arrays.
## - The second section plots data read from a text file.
##
## \macro_image
## \macro_code
##
## \author Rene Brun, Jamie Gooding

import re

import numpy as np
import ROOT

c1 = ROOT.TCanvas("c1", "Two simple graphs", 200, 10, 700, 500)
c1.Divide(
    2, 1
)  # Dividing the canvas in subpads for distinguishing the two examples, [See documentation](https://root.cern/doc/master/classTCanvas.html)

# First Example (Available data)
c1.cd(1)

n = 20
x = np.linspace(0, n - 1, n)
y = 10 * np.sin(x + 0.2)

gr1 = ROOT.TGraph(n, x, y)  # Create a TGraph object, storing the number of data n and the x, y variables

# Set the color, width and style for the markers and line

gr1.SetLineColor(2)
gr1.SetLineWidth(4)
gr1.SetMarkerColor(4)
gr1.SetMarkerStyle(21)
gr1.SetTitle("Graph from available data")  # Choose title for the graph
gr1.GetXaxis().SetTitle("X title")  # Choose title for the axis
gr1.GetYaxis().SetTitle("Y title")

# Uncomment the following line to set a custom range for the x-axis (respectively for the y-axis):
# gr1.GetXaxis().SetRangeUser(0, 1.8)

gr1.Draw(
    "ACP"
)  # "A" draw axes, "C" = draw a smooth line through the markers (optional) and "P" = draw markers for data points
# Optional customization can be done on a ROOT interactive session


# SECOND EXAMPLE (Data stored in a text file)
c1.cd(2)

w = np.array([])
z = np.array([])

# Open the data file
with open(f"{ROOT.gROOT.GetTutorialDir()}/visualisation/graphs/data_basic.txt") as file:  # Open the data file
    for line in file:
        w_str, z_str = re.split(r"\s+", line)[:2]
        w = np.append(w, float(w_str))
        z = np.append(z, float(z_str))

m = len(w)

gr2 = ROOT.TGraph(m, w, z)  # Create a TGraph object for the file data
gr2.SetLineColor(4)
gr2.SetLineWidth(2)
gr2.SetMarkerColor(2)
gr2.SetMarkerStyle(20)
gr2.SetTitle("Graph from data file")
gr2.GetXaxis().SetTitle("W title")
gr2.GetYaxis().SetTitle("Z title")

gr2.Draw("ACP")
