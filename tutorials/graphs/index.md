\addtogroup tutorial_graphs

@{
A [TGraph](https://root.cern/doc/master/classTGraph.html) is an object made of two arrays X and Y with npoints each. ROOT can create TGraphs automatically by reading a text file (e.g. CSV), and other methods exist to create them from a function, from another TGraph, a histogram, vectors, arrays, or even by adding pairs of points one by one.

In addition, ROOT offers multiple variations to basic TGraphs, such as:
- [TGraph2D](https://root.cern/doc/master/classTGraph2D.html)
- [TGraph2DAsymmErrors](https://root.cern/doc/master/classTGraph2DAsymmErrors.html)
- [TGraph2DErrrors](https://root.cern/doc/master/classTGraph2DErrors.html)
- [TGraphAsymmErrors](https://root.cern/doc/master/classTGraphAsymmErrors.html)
- [TGraphBentErrors](https://root.cern/doc/master/classTGraphBentErrors.html)
- [TGraphDelaunay](https://root.cern/doc/master/classTGraphDelaunay.html)
- [TGraphDelaunay2D](https://root.cern/doc/master/classTGraphDelaunay2D.html)
- [TGraphErrors](https://root.cern/doc/master/classTGraphErrors.html)
- [TGraphMultiErrors](https://root.cern/doc/master/classTGraphMultiErrors.html)
- [TGraphPolar](https://root.cern/doc/master/classTGraphPolar.html)
- [TGraphQQ](https://root.cern/doc/master/classTGraphQQ.html)
- [TGraphSmooth](https://root.cern/doc/master/classTGraphSmooth.html)
- [TGraphStruct](https://root.cern/doc/master/classTGraphStruct.html)
- [TGraphTime](https://root.cern/doc/master/classTGraphTime.html)

TGraphs are painted through the [TGraphPainter](https://root.cern/doc/master/classTGraphPainter.html) or [TGraph2DPainter](https://root.cern/doc/master/classTGraph2DPainter.html) class. As a general remark, TGraphs are not binned, each point is painted individually, and when the "A" draw option is used, ROOT draws a histogram, which can be retrieved with `graph->GetHistogram()` after drawing it, to display a frame with x and y axes. Note that the bins of this histogram do not correspond to the points of the graph (i.e., generally speaking bin 1 is not the same as the x value of point 1, and so on).

To draw two or more TGraphs superposed, there are two basic approaches:
1. Draw the first one with the option "A" included (so that the axes are drawn), and successively draw the next one/s (without the "A" option).
2. Create a [TMultiGraph](https://root.cern/doc/master/classTMultiGraph.html) and Add all graphs to it, then draw the `TMultiGraph`.

In the second approach, `TMultiGraph` automatically sets an appropriate scale for the plot so that all graphs are shown. In the first approach, the scale is determined by the graph that was drawn with "A", which may or may not be large enough for all graphs.

The graph tutorials below are divided in groups of increasing complexity, starting with examples showing how to create, fill, and draw the different types of graphs.

## Tutorials sorted after groups
- [Basics: creation and drawing](\ref basics)
- [Formatting: changing/adding elements to the graphs and/or the plots](\ref modfying)
- [Intermediate: more advanced examples](\ref medium)
- [More tutorials](\ref other)


[List of all tutorials](\ref alltutorials)
\anchor basics
## Basics

These examples showscase the creation of different types of graphs and basic ways to plot them.

| **Tutorial** || **Description** |
|------|--------|-----------------|
| gr001_basic.C |  | Create a simple graph from available data or from a file, and draw it. |
| gr002_err_1g.C |  | Create and draw a graph with error bars.|
| gr003_err_2gr.C |  | Create and draw two graphs with error bars, superposed on the same canvas (not using TMultiGraph). |
| gr004_err_asym.C |  | Create and draw a graph with asymmetric x & y errors. |
| gr005_apply.C |  | Demonstrate the functionality of the TGraph::Apply() method. |
| gr006_scatter.C |  | Scatter plot for 4 variables, mapped to: x, y, marker colour and marker size. |
| gr007_multigraph.C |  | Create and draw a TMultiGraph (several graphs superposed). |
| gr008_multierrors.C |  | Graph with multiple y errors in each bin. |
| gr009_bent_err.C | gr009_bent_err.py | Graph with bent (non-vertical/non-horizontal) error bars. |
| gr010_approx_smooth.C |  | Create a TGraphSmooth and show the usage of the interpolation function Approx. |
| gr011_2Derrorsfit.C |  | Create, draw and fit a TGraph2DErrors. |
| gr012_polar.C |  | Create and draw a polar graph. |
| gr013_polar2.C |  | Polar graph with errors and polar axis in radians (PI fractions). |
| gr014_polar3.C |  | Create a polar graph using a TF1 and draw it with PI axis. |
| gr015_smooth.C |  | Show scatter plot smoothers: ksmooth, lowess, supsmu |
| gr016_struct.C |  | Draw a simple graph structure. |
| gr017_time.C |  | Example of TGraphTime. |
| gr018_time2.C |  | TGraphTime to visualise a set of particles with their time stamp in a MonteCarlo program. |


\anchor modifying
## Formatting

These examples show

| **Tutorial** || **Description** |
|------|--------|-----------------|
| gr101_shade_area.C |  | Shows how to shade an area between two graphs. |
| gr102_reverse_axis.C |  | How to reverse the points of a graph along x. |
| gr103_zones.C | gr103_zones.py | How to divide a canvas into adjacent subpads, with axis labels on the top and right side of the pads. |
| gr104_palettecolor.C |  | Automatically set graph colours from a palette. |
| gr105_multigraphpalettecolor.C |  | Automatically set multi-graph colours from a palette. |
| gr106_exclusiongraph.C |  | Draw three graphs (in a TMultiGraph) with exclusion zones. |
| gr107_exclusiongraph2.C |  | Draw graphs (superposed, but no TMultiGraph) with exclusion zones. |
| gr108_timeSeriesFromCSV.C | gr108_timeSeriesFromCSV.py | Use of the time axis on a TGraph
 with data read from a text file. |
| gr109_timeSeriesFromCSV_TDF.C |  | Use of the time axis on a TGraph with data read from a text file, but using [RDataFrame](https://root.cern/doc/master/classROOT_1_1RDataFrame.html). |


\anchor medium
## Intermediate

These examples show

| **Tutorial** || **Description** |
|------|--------|-----------------|
| gr201_waves.C |  | Draw spherical waves interference, using closed and filled TGraphs to hide other plot elements. |
| gr202_textmarkers.C |  | Draw a graph with text attached to each point. Uses a [TExec](https://root.cern/doc/master/classTExec.html) function to attach the text to the points. |
|  |  |  |


\anchor other
## More tutorials

These examples show

| **Tutorial** || **Description** |
|------|--------|-----------------|
| gr301_highlight1.C |  | How to use the interactive highlight mode on graph, thanks to the TCanvas [HighlightConnect](https://root.cern/doc/master/classTCanvas.html#a462b8dc286a2d29152fefa9b31f89920) method. |
| gr302_highlight2.C |  | How to use the interactive highlight mode on graph, thanks to the TCanvas [HighlightConnect](https://root.cern/doc/master/classTCanvas.html#a462b8dc286a2d29152fefa9b31f89920) method. |
| gr303_zdemo.C |  | Example of graphs in log scales with annotations and other advanced plot formatting. |


\anchor alltutorials
@}