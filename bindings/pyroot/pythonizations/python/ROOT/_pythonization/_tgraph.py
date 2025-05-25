# Author: Enric Tejedor CERN  03/2019
# Authors of the documentaion: Vincenzo Eduardo Padulano CERN, Marta Adamina Krawczyk CERN 11/2024

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
\pythondoc TGraph

The `TGraph` class in PyROOT is a versatile tool for creating and visualizing 2D graphs with points connected by straight lines. It is commonly used for plotting scatter plots, line graphs, and interpolated data visualizations.

### Key Features:
1. **Create and manipulate 2D graphs** with dynamic data points.
2. **Customize graph appearance:** markers, lines, colors, and more.
3. **Add labels, legends, and multiple datasets** for comprehensive visualization.
4. **Save graphs** in formats like .png, .pdf, and .root.

Below are two examples:
1. **Basic Example:** Demonstrating TGraph functionalities.
2. **Advanced Example:** Visualizing and comparing multiple datasets.

## Basic Example Usage

\code{.py}
import ROOT
import numpy as np

# Create a canvas to draw the graph
canvas = ROOT.TCanvas("c1", "TGraph Example", 800, 600)

# Define the number of data points
n = 10
# Generate X values as a sequence from 0 to 9
x = numpy.linspace(0, 9, n)
# Compute Y values as the square of X values
y = x**2

# Create a TGraph object using the defined X and Y values
graph = ROOT.TGraph(n, x, y)

# Get the current number of points in the graph
n = graph.GetN()
# Add a new point to the graph at the specified index
graph.SetPoint(n, 10, 100)

# Retrieve the coordinates of the 3rd point (index 2)
x_val = graph.GetPointX(2)
y_val = graph.GetPointY(2)
print(f"Point {2}: ({x_val}, {y_val})")

# Customize graph appearance
graph.SetTitle("Example Graph;X-axis Title;Y-axis Title")
graph.SetMarkerStyle(21)
graph.SetMarkerSize(2)
graph.SetMarkerColor(4)
graph.SetLineColor(2)
graph.SetLineStyle(1)
graph.SetLineWidth(2)

# Draw the graph with axes ("A"), lines ("L"), and points ("P")
graph.Draw("ALP")

# Save the graph as a PNG file
canvas.SaveAs("tgraph_example.png")
\endcode

## Advanced Example: Comparing Two Datasets
This example demonstrates how to visualize and compare multiple datasets using `TGraph`. The script customizes axes, labels and legends. The output is a comparative graph saved as a PDF.

\code{.py}
import ROOT
import numpy

# Define benchmark data for Dataset 1 using NumPy arrays
dataset1_x_values = numpy.array([32, 64, 128, 256, 512, 1024, 2048], dtype=float)
dataset1_y_values = numpy.array([2, 3, 7, 13, 26, 48, 100], dtype=float)

# Define benchmark data for Dataset 2 using NumPy arrays
dataset2_x_values = numpy.array([32, 64, 128, 256, 512, 1024, 2048], dtype=float)
dataset2_y_values = numpy.array([2, 3, 7, 13, 26, 47, 92], dtype=float)

# Create TGraph objects for Dataset 1 and Dataset 2
graph1 = ROOT.TGraph(len(dataset1_x_values), dataset1_x_values, dataset1_y_values)
graph2 = ROOT.TGraph(len(dataset2_x_values), dataset2_x_values, dataset2_y_values)

# Configure canvas dimensions for displaying the comparative plot
width, height = 1920, 1080
canvas = ROOT.TCanvas("c", "Datasets Comparison", width, height)
# Left, Right, Bottom, Top margins
canvas.SetMargin(0.08, 0.05, 0.15, 0.09)

# Retrieve y-axis minimum and maximum values to set dynamic axis ranges for the plot
ymin = graph1.GetHistogram().GetMinimum()
ymax = graph1.GetHistogram().GetMaximum()

# Configure graph1
graph1.SetTitle("TGraph dataset comparison plot title")
graph1.GetXaxis().SetTitle("X Axis Title")
graph1.GetXaxis().SetTitleSize(0.04)
graph1.GetXaxis().SetTitleOffset(1.3)
graph1.GetXaxis().SetLabelSize(0.03)

# Hide default x-axis labels and ticks to customize axis appearance
graph1.GetXaxis().SetLabelSize(0)
graph1.GetXaxis().SetTickLength(0)

graph1.GetYaxis().SetTitle("Y Axis Title")
graph1.GetYaxis().SetTitleSize(0.04)
graph1.GetYaxis().SetTitleOffset(0.7)
graph1.GetYaxis().SetLabelSize(0.03)
# Ensure y-axis includes zero
graph1.GetYaxis().SetRangeUser(0, ymax)

graph1.SetMarkerColor(ROOT.kAzure - 3)
graph1.SetMarkerStyle(20)
graph1.SetMarkerSize(2)
graph1.SetLineColor(ROOT.kAzure - 3)
graph1.SetLineWidth(2)

# Draw graph1
graph1.Draw("ALP")

# Add custom x-axis labels and ticks to align with data points for better readability
dy = ymax - ymin
xaxislabels = []
xaxisticks = []

for i in range(len(dataset1_x_values)):
    # Get the x-coordinate of the i-th data point
    x = graph1.GetPointX(i)
    
    # Add custom label for the x-axis
    # Place the label slightly below the x-axis
    label = ROOT.TLatex(x, -8, str(int(x)))
    label.SetTextSize(0.03)
     # Use a standard ROOT font         
    label.SetTextFont(42)
    # Align text to the left and bottom relative to the anchor point
    label.SetTextAlign(11)
    # Rotate the label 45 degrees for better readability
    label.SetTextAngle(45)
    label.Draw()
    # Keep a reference to the label to ensure it persists on the canvas               
    xaxislabels.append(label)
    
    # Create a tick at the x-coordinate
    # Draw a tick line below the axis
    tick = ROOT.TLine(x, 0, x, 0 - 0.02 * dy)
    tick.Draw()
    # Keep a reference to the tick to ensure it persists on the canvas                    
    xaxisticks.append(tick)

# Configure graph2
graph2.SetMarkerColor(ROOT.kOrange + 1)
graph2.SetMarkerStyle(43)
graph2.SetMarkerSize(4)
graph2.SetLineColor(ROOT.kOrange + 1)
graph2.SetLineWidth(2)

# Draw graph2
graph2.Draw("SAME LP")

# Add a legend to distinguish Dataset 1 and Dataset 2
legend = ROOT.TLegend(0.2, 0.75, 0.4, 0.91)
legend.AddEntry(graph1, "Dataset 1", "lp")
legend.AddEntry(graph2, "Dataset 2", "lp")
legend.SetTextSize(0.03)
legend.Draw()

# Save the canvas as a PDF
canvas.SaveAs("tgraph_comparison.pdf")
\endcode

\endpythondoc
'''

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

