\addtogroup tutorial_hist

@{
A histogram is a visual representation of the distribution of quantitative data.
[TH1 class](classTH1.html) introduces the basic data structure used in root for histograms.

In a nutshell:

~~~{.cpp}
  // Open the file to write the histogram to
   auto outFile = std::unique_ptr<TFile>(TFile::Open("outfile.root", "RECREATE"));

   // Create the histogram object
   // There are several constructors you can use (\see TH1). In this example we use the
   // simplest one, accepting a number of bins and a range.
   int nBins = 30;
   double rangeMin = 0.0;
   double rangeMax = 10.0;
   TH1D histogram("histogram", "My first ROOT histogram", nBins, rangeMin, rangeMax);

   // Fill the histogram. In this simple example we use a fake set of data.
   // The 'D' in TH1D stands for 'double', so we fill the histogram with doubles.
   // In general you should prefer TH1D over TH1F unless you have a very specific reason
   // to do otherwise.
   const std::array values{1, 2, 3, 3, 3, 4, 3, 2, 1, 0};
   for (double val : values) {
      histogram.Fill(val);
   }

   // Write the histogram to `outFile`.
   outFile->WriteObject(&histogram, histogram.GetName());

   // When the TFile goes out of scope it will close itself and write its contents to disk.
~~~

Explore the examples below for [different histogram classes](group__Histograms.html)

## Tutorials sorted after groups

- [Introduction](\ref introduction)
- [TH1 Histograms](\ref th1)
- [TH2 Histograms](\ref th2)
- [THStack](\ref thstack)
- [TRatio plots](\ref ratioplots)
- [TPoly](\ref tpoly)
- [Graphics](\ref graphics)
- [TExec](\ref texec)

[List of all tutorials](\ref alltutorials)
\anchor introduction

## Introduction

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
| hist000_TH1_first.C      | First example              |

\anchor th1

## TH1 Histograms

These examples shows some of the ratioplots

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist000_TH1_first.C|Hello World example for TH1|
|hist001_TH1_fillrandom.C|Fill a 1D histogram with random values using predefined functions|
|hist002_TH1_fillrandom_userfunc.C|Fill a 1D histogram from a user-defined parametric function.|
|hist003_TH1_draw.C|Draw a 1D histogram to a canvas.|
|hist004_TH1_labels.C|1D histograms with alphanumeric labels.|
|hist005_TH1_palettecolor.C|Palette coloring for TH1|
|hist006_TH1_bar_charts.C|Draw 1D histograms as bar charts|
|hist007_TH1_liveupdate.C|Histograms filled and drawn in a loop.|
|hist008_TH1_zoom.C|Changing the Range on the X-Axis of a Histogram|
|hist009_TH1_normalize.C|Normalizing a Histogram|
|hist010_TH1_two_scales.C|Example of macro illustrating how to superimpose two histograms|
|hist011_TH1_legend_autoplaced.C|The legend can be placed automatically in the current pad in an empty space|
|hist012_TH1_hksimple.C|Illustrates the advantages of a TH1K histogram|
|hist013_TH1_rebin.C|Rebin a variable bin-width histogram.|
|hist014_TH1_cumulative.C|Illustrate use of the TH1::GetCumulative method.|
|hist015_TH1_read_and_draw.C|Read a 1-D histogram from a ROOT File and draw it.|
|hist016_TH1_different_scales_canvas.C|Example of a canvas showing two histograms with different scales.|
|hist017_TH1_smooth.C|Histogram smoothing.|
|hist101_TH1_autobinning.C|Fill multiple histograms with different functions and automatic binning.|
|hist060_Stats.C|Edit statistics box.|

\anchor th2

## TH2 Histograms

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist018_TH2_cutg.C|This example demonstrates how to display a 2D histogram and|
|hist019_TH2_projection.C|This example demonstrates how to display a histogram and its two projections.|
|hist020_TH2_draw.C|Display the various 2-d drawing options|
|hist021_TH2_reverse_axis.C|Example showing an histogram with reverse axis.|
|hist022_TH2_palette.C|When an histogram is drawn with the option `COLZ`, a palette is automatically drawn|
|hist102_TH2_contour_list.C|Getting Contours From TH2D.|

\anchor thstack

## THStack

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist023_THStack_simple.C|Example of stacked histograms: class THStack.|
|hist024_THStack_pads.C|Drawing stack histograms on subpads.|
|hist025_THStack_2d_palette_color.C|Palette coloring for 2D histograms' stack is activated thanks to the option `PFC`|
|hist026_THStack_color_scheme.C|This example demonstrates how to use the accessible color schemes with THStack.|
|hist027_THStack_palette_color.C|Palette coloring for histograms' stack is activated thanks to the options `PFC`|
|hist028_THStack_multicolor.C|Use a THStack to show a 2-D hist with cells with different colors.|

\anchor ratioplots

## Ratio plots

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist029_TRatioPlot_simple.C|Example creating a simple ratio plot of two histograms using the `pois` division option.|
|hist030_TRatioPlot_residual.C|Example of a fit residual plot.|
|hist031_TRatioPlot_residual_fit.C|Example which shows how you can get the graph of the lower plot and set the y axis range for it.|
|hist032_TRatioPlot_fit_lines.C|Example that shows custom dashed lines on the lower plot, specified by a vector of floats.|
|hist033_TRatioPlot_fit_confidence.C|Example that shows how you can set the colors of the confidence interval bands by using|
|hist034_TRatioPlot_fit_margin.C|Example showing a fit residual plot, where the separation margin has been set to 0.|
|hist035_TRatioPlot_manual_ratio.C|Example displaying two histograms and their ratio. This macro does not use the|

\anchor tpoly

## TPoly

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist037_TH2Poly_boxes.C|This tutorial illustrates how to create an histogram with polygonal|
|hist038_TH2Poly_honeycomb.C|This tutorial illustrates how to create an histogram with hexagonal|
|hist039_TH2Poly_usa.C|This tutorial illustrates how to create an histogram with polygonal|
|hist040_TH2Poly_europe.C|This tutorial illustrates how to create an histogram with polygonal|
|hist041_TProfile2Poly_realistic.C|Different charges depending on region|
|hist042_TProfile2Poly_module_error.C|Simulate faulty detector panel w.r.t. particle charge|
|hist056_TPolyMarker_contour.C|Make a contour plot and get the first contour in a TPolyMarker.|
|hist104_TH2Poly_fibonacci.C|A TH2Poly build with Fibonacci numbers.|

\anchor graphics

## Graphics

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist043_Graphics_highlight.C|This tutorial demonstrates how the highlight mechanism can be used on an histogram.|
|hist044_Graphics_highlight2D.C|This tutorial demonstrates how the highlight mechanism can be used on an histogram.|
|hist045_Graphics_highlight_ntuple.C|This tutorial demonstrates how the highlight mechanism can be used on a ntuple.|
|hist046_Graphics_highlight1D.C|This tutorial demonstrates how the highlight mechanism can be used on an histogram.|
|hist047_Graphics_candle_decay.C|Candle Decay, illustrate a time development of a certain value.|
|hist048_Graphics_candle_hist.C|Example showing how to combine the various candle plot options.|
|hist049_Graphics_candle_plot.C|Example of candle plot with 2-D histograms.|
|hist050_Graphics_candle_plot_options.C|Example showing how to combine the various candle plot options.|
|hist051_Graphics_candle_plot_stack.C|Example showing how a THStack with candle plot option.|
|hist052_Graphics_candle_plot_whiskers.C|Example of candle plot showing the whiskers definition.|
|hist053_Graphics_candle_scaled.C|Candle Scaled, illustrates what scaling does on candle and violin charts.|
|hist054_Graphics_logscales.C|Draw parametric functions with log scales.|
|hist055_Graphics_xyplot.C|Example showing how to produce a plot with an orthogonal axis system|

\anchor texec

## TExec

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist057_TExec_th1.C|Echo object at mouse position.|
|hist058_TExec_th2.C|Echo object at mouse position and show a graphics line.|
|hist105_TExec_dynamic_slice.C|Show the slice of a TH2 following the mouse position.|

\anchor alltutorials

@}
