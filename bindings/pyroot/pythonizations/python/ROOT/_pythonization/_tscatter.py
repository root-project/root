# Author: Marta Adamina Krawczyk CERN 11/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
\pythondoc TScatter

The `TScatter` class in PyROOT can be used to create and visualize scatter plots dynamically. By integrating with NumPy, it allows efficient data management and customization.

### Key Features:
1. **Efficient Data Handling**: Use NumPy arrays to manage and pass data points to the `TScatter` object.
2. **Visual Customization**: Adjust marker styles, colors, and titles to tailor the visualization.
3. **Canvas Layout Control**: Configure canvas margins to ensure that titles and axes are displayed without clipping.
4. **Exporting Plots**: Save scatter plots in various formats (e.g., `.png`, `.pdf`) for further use.

### Example Usage:
Below is a Python example demonstrating the use of `TScatter`:

\code{.py}
import ROOT
import numpy as np

# Create a TCanvas and configure margins
canvas = ROOT.TCanvas("c1", "Scatter Plot Example", 800, 600)
# Margins (left, right, bottom, top) to adjust the canvas layout and prevent clipping of titles or axis labels.
canvas.SetMargin(0.1, 0.13, 0.1, 0.1)

# Define the number of data points
n = 175

# Generate arrays for x, y coordinates and optional metadata
x = 100*np.random.rand(n)
y = 200*np.random.rand(n)
c = 300*np.random.rand(n)  # Optional: Color or other metadata
s = 400*np.random.rand(n)  # Optional: Size or other metadata

# Create a TScatter object with the data
scatter = ROOT.TScatter(n, x, y, c, s)

# Customize the plot's appearance
scatter.SetMarkerStyle(20) # Set marker style
scatter.SetMarkerColor(4)  # Set marker color
scatter.SetTitle("Scatter plot title;X title;Y title;Z title")  # Set plot and axis titles
scatter.GetZaxis().SetTitleSize(0.04)  # Modify the title size
scatter.GetZaxis().SetTitleOffset(1.0)  # Adjust the z-axis title offset

# Draw the scatter plot
scatter.Draw("AP")  # "A" for Axis, "P" for Points

# Save the scatter plot to a file
canvas.SaveAs("scatter_plot_example.png")
\endcode

\endpythondoc
'''
