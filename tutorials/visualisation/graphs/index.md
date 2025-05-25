\addtogroup tutorial_graphs

@{
A [TGraph](https://root.cern/doc/master/classTGraph.html) is an object made of two arrays X and Y with n points each. ROOT can create TGraphs automatically by reading a text file (e.g. CSV), using a function, another TGraph, histogram, vectors, arrays, or even by adding pairs of points one by one.

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
- [TGraphTime](https://root.cern/doc/master/classTGraphTime.html).

TGraphs are painted through the [TGraphPainter](https://root.cern/doc/master/classTGraphPainter.html) or [TGraph2DPainter](https://root.cern/doc/master/classTGraph2DPainter.html) classes. As a general remark, TGraphs are not binned, each point is painted individually.

The graph tutorials below are divided in groups of increasing complexity, starting with examples showing how to create, fill, and draw the different types of graphs.

## Tutorials sorted after groups
- [Basics: creation and drawing](\ref basics)
- [Formatting: changing/adding elements to the graphs and/or the plots](\ref modfying)
- [Intermediate: more advanced examples](\ref medium)
- [More tutorials](\ref graph_other)


[List of all tutorials](\ref graphs_alltutorials)
\anchor basics
## Basics

These examples showcase the creation of different types of graphs and basic ways to plot them.

| **Tutorial** || **Description** |
|---------------------------|-------------------------------|-----------------|
| gr001_simple.C            | gr001_simple.py               | Create a simple graph from available data or from a file, and draw it. |
| gr002_errors.C            | gr002_errors.py               | Create and draw a graph with error bars.|
| gr003_errors2.C           | gr003_errors2.py              | Create and draw two graphs with error bars, superposed on the same canvas (not using TMultiGraph). |
| gr004_errors_asym.C       | gr004_errors_asym.py          | Create and draw a graph with asymmetric x & y errors. |
| gr005_apply.C             | gr005_apply.py                | Demonstrate the functionality of the TGraph::Apply() method. |
| gr006_scatter.C           | gr006_scatter.py              | Scatter plot for 4 variables, mapped to: x, y, marker color and marker size. |
| gr007_multigraph.C        | gr007_multigraph.py           | Create and draw a TMultiGraph (several graphs superposed). |
| gr008_multierrors.C       |                               | Graph with multiple y errors in each bin. |
| gr009_bent_err.C          | gr009_bent_err.py             | Graph with bent (non-vertical/non-horizontal) error bars. |
| gr010_approx_smooth.C     | gr010_approx_smooth.py        | Create a TGraphSmooth and show the usage of the interpolation function Approx. |
| gr011_graph2d_errorsfit.C | gr011_graph2d_errorsfit.py    | Create, draw and fit a TGraph2DErrors. |
| gr012_polar.C             | gr012_polar.py                | Create and draw a polar graph. |
| gr013_polar2.C            | gr013_polar2.py               | Polar graph with errors and polar axis in radians (PI fractions). |
| gr014_polar3.C            | gr014_polar3.py               | Create a polar graph using a TF1 and draw it with PI axis. |
| gr015_smooth.C            |                               | Show scatter plot smoothers: ksmooth, lowess, supsmu |
| gr016_struct.C            |                               | Draw a simple graph structure. |
| gr017_time.C              |                               | Example of TGraphTime. |
| gr018_time2.C             |                               | TGraphTime to visualize a set of particles with their time stamp in a MonteCarlo program. |


\anchor modifying
## Formatting

These examples show different ways of formatting the graphs, in particular how to:

| **Tutorial** || **Description** |
|------|--------|-----------------|
| gr101_shade_area.C |  |  Shade an area between two graphs. |
| gr102_reverse_graph.C |  | Reverse points of a graph along x. |
| gr103_zones.C | gr103_zones.py | Divide a canvas into adjacent subpads, with axis labels on the top and right side of the pads. |
| gr104_palettecolor.C |  | Automatically set graph colors from a palette. |
| gr105_multigraphpalettecolor.C |  | Automatically set multi-graph colors from a palette. |
| gr106_exclusiongraph.C |  | Draw three graphs (in a TMultiGraph) with exclusion zones. |
| gr107_exclusiongraph2.C |  | Draw graphs (superposed, but no TMultiGraph) with exclusion zones. |
| gr108_timeSeriesFromCSV.C | gr108_timeSeriesFromCSV.py | Use of the time axis on a TGraph with data read from a text file. |
| gr109_timeSeriesFromCSV_RDF.C |  | Use of the time axis on a TGraph with data read from a text file, but using [RDataFrame](https://root.cern/doc/master/classROOT_1_1RDataFrame.html). |
| gr110_logscale.C |  | Set logarithmic scale for the axes of the graph.|
| gr111_legend.C |  | Add a legend.|
| gr112_reverse_graph_and_errors.C |  | Reverse points of the graph along both axes - examples with various error types. |

\anchor medium
## Intermediate

These examples are slightly more advanced so will be most useful for more advanced users.

| **Tutorial**        | **Description**  |
|---------------------|------------------|
| gr201_waves.C       | Draw spherical waves interference, using closed and filled TGraphs to hide other plot elements. |
| gr202_textmarkers.C | Draw a graph with text attached to each point. Uses a [TExec](https://root.cern/doc/master/classTExec.html) function to attach the text to the points. |


\anchor graph_other
## More tutorials

These examples show the most complex usage of the graphs functionality.

| **Tutorial**       | **Description**  |
|--------------------|------------------|
| gr301_highlight1.C |  How to use the interactive highlight mode on graph, thanks to the TCanvas [HighlightConnect](https://root.cern/doc/master/classTCanvas.html#a462b8dc286a2d29152fefa9b31f89920) method. |
| gr302_highlight2.C |  How to use the interactive highlight mode on graph, thanks to the TCanvas [HighlightConnect](https://root.cern/doc/master/classTCanvas.html#a462b8dc286a2d29152fefa9b31f89920) method. |
| gr303_zdemo.C      |  Example of graphs in log scales with annotations and other advanced plot formatting. |


\anchor graphs_alltutorials
@}