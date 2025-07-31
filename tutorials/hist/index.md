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

- [Introduction](\ref hist_intro)
- [TH1 Histograms](\ref th1)
- [TH2 Histograms](\ref th2)
- [THnSparse](\ref thnsparse)
- [THStack](\ref thstack)
- [TRatio plots](\ref ratioplots)
- [TPoly](\ref tpoly)
- [Graphics](\ref hist_graphics)
- [TExec](\ref texec)

[List of all tutorials](\ref hist_alltutorials)
\anchor hist_intro

## Introduction

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
| hist000_TH1_first.C      |First example.              |

\anchor th1


## TH1 Histograms
The examples below showcase the same functionalities through 3 different implementations: the first column is a C++ macro, the second column corresponds to the Python implementation, and the third column is a Python implementation showcasing the newest Pythonizations available in ROOT.


## Tutorial
|                                    |           **Tutorial**          |                               |     **Description**    |
|------------------------------------|---------------------------------|-------------------------------|------------------------|
|**C++ macro** |**Python** |**Python+Newest Pythonization**||
| hist000_TH1_first.C                | hist000_TH1_first.py            |   hist000_TH1_first_uhi.py    |Hello World example for TH1. |
| hist001_TH1_fillrandom.C           | hist001_TH1_fillrandom.py       |  hist001_TH1_fillrandom_uhi.py    | Fill a 1D histogram with random values using predefined functions. |
| hist002_TH1_fillrandom_userfunc.C  | hist002_TH1_fillrandom_userfunc.py |hist002_TH1_fillrandom_userfunc_uhi.py| Fill a 1D histogram from a user-defined parametric function. |
| hist003_TH1_draw.C                 | hist003_TH1_draw.py              | hist003_TH1_draw_uhi.py|Draw a 1D histogram to a canvas. |
| hist004_TH1_labels.C               |                                  |                              | 1D histograms with alphanumeric labels. |
| hist005_TH1_palettecolor.C         |                                  |                              | Palette coloring for TH1. |
| hist006_TH1_bar_charts.C           |                                  |                              | Draw 1D histograms as bar charts. |
| hist007_TH1_liveupdate.C           | hist007_TH1_liveupdate.py        |hist007_TH1_liveupdate_uhi.py | Histograms filled and drawn in a loop. |
| hist008_TH1_zoom.C                 |                                  |                               | Change the range of an axis in a Histogram. |
| hist009_TH1_normalize.C            |                                  |                               | Normalizing a Histogram. |
| hist010_TH1_two_scales.C           | hist010_TH1_two_scales.py        | hist010_TH1_two_scales_uhi.py|Draw two histograms on one canvas using different y-axis scales. |
| hist011_TH1_legend_autoplaced.C    |                                  |                               | Automatic placing of the legend. |
| hist012_TH1_hksimple.C             |                                  |                               | Dynamic filling of TH1K histograms. |
| hist013_TH1_rebin.C                |                                  |                               | Create a variable bin-width histogram and change bin sizes. |
| hist014_TH1_cumulative.C           |                                  |                               | Illustrate use of the TH1::GetCumulative method. |
| hist015_TH1_read_and_draw.C        | hist015_TH1_read_and_draw.py     | hist015_TH1_read_and_draw_uhi.py|Read a 1D histogram from a ROOT File and draw it. |
| hist016_TH1_different_scales_canvas.C |                               |                               | Draw two histograms on one canvas using different y-axis scales. |
| hist017_TH1_smooth.C               |                                  |                               | Histogram smoothing. |
| hist060_TH1_stats.C                |                                  |                               | Edit statistics box. |
| hist061_TH1_timeonaxis.C           |                                  |                               | Use a time axis as an x axis. |
| hist062_TH1_timeonaxis2.C          |                                  |                               | Use a time axis as an x axis and use a time offset. |
| hist063_TH1_seism.C                |                                  |                               | Use a time axis as an x axis to show sine signal as a strip chart. |
| hist101_TH1_autobinning.C          |                                  |                               | Fill multiple histograms with different functions and automatic binning. |

\anchor th2

## TH2 Histograms

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist018_TH2_cutg.C|Use TCutG object to select bins for drawing a region of a 2D histogram.|
|hist019_TH2_projection.C|Display a histogram and its two projections.|
|hist020_TH2_draw.C|Display 2D histogram drawing options.|
|hist021_TH2_reverse_axis.C|Histogram with reverse axis.|
|hist022_TH2_palette.C|Automatic placing of a color palette via option `COLZ`.|
|hist102_TH2_contour_list.C|Get contours from a 2D histogram.|

\anchor thstack

## THnSparse
| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist103_THnSparse_hist.C|Evaluate the performance of THnSparse vs TH1/2/3/n.|

\anchor thnsparse

## THStack

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist023_THStack_simple.C|Stack histograms with class THStack.|
|hist024_THStack_pads.C|Draw stack histograms on subpads.|
|hist025_THStack_2d_palette_color.C|Display multiple 2D histograms picking colors within palette 1.|
|hist026_THStack_color_scheme.C|Use accessible color schemes with THStack.|
|hist027_THStack_palette_color.C|Display multiple 1D histograms picking colors within palette kOcean.|
|hist028_THStack_multicolor.C|Use a THStack to show a 2D histogram with cells with different colors.|

\anchor ratioplots

## Ratio plots

|                                    |           **Tutorial**          |                               |     **Description**    |
|------------------------------------|---------------------------------|-------------------------------|------------------------|
|**C++ macro** |**Python** |**Python+Newest Pythonization**| |
|hist029_TRatioPlot_simple.C|hist029_TRatioPlot_simple.py||Create a simple ratio plot of two histograms using the `pois` division option.|
|hist030_TRatioPlot_residual.C|hist030_TRatioPlot_residual.py|hist030_TRatioPlot_residual_uhi.py |Create a fit residual plot.|
|hist031_TRatioPlot_residual_fit.C|hist031_TRatioPlot_residual_fit.py||Create a fit residual plot and set the y-axis range for it.|
|hist032_TRatioPlot_fit_lines.C|hist032_TRatioPlot_fit_lines.py||Set custom dashed lines specified by a vector of floats.|
|hist033_TRatioPlot_fit_confidence.C|hist033_TRatioPlot_fit_confidence.py||Set the colors of the confidence interval bands by using|
|hist034_TRatioPlot_fit_margin.C|hist034_TRatioPlot_fit_margin.py||Create a fit residual plot, where the separation margin has been set to 0.|



\anchor tpoly

## TPoly

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist037_TH2Poly_boxes.C|Create a histogram with polygonal bins (TH2Poly).|
|hist038_TH2Poly_honeycomb.C|Create a histogram with hexagonal bins (TH2Poly).|
|hist039_TH2Poly_usa.C|Create a histogram with polygonal bins (TH2Poly). The initial data represent the USA map.|
|hist040_TH2Poly_europe.C|Create a histogram with polygonal bins (TH2Poly). The initial data represent the Europe map.|
|hist041_TProfile2Poly_realistic.C|Create a histogram with polygonal bins representing different particle charges in a detectior.|
|hist042_TProfile2Poly_module_error.C|Create a histogram with polygonal bins simulating a faulty detector panel w.r.t. particle charge.|
|hist056_TPolyMarker_contour.C|Make a contour plot and get the first contour in a TPolyMarker.|
|hist104_TH2Poly_fibonacci.C|Create a histogram representing the "Fibonacci spiral".|

\anchor hist_graphics

## Graphics

| **Tutorial**             |    **Description**         |
|--------------------------|----------------------------|
|hist043_Graphics_highlight.C|Use the highlight mechanism to update the title of a histogram in real time.|
|hist044_Graphics_highlight2D.C|Use the highlight mechanism to displaying the X and Y projections at a bin in real time.|
|hist045_Graphics_highlight_ntuple.C|Use the highlight mechanism to display which events of an ntuple contribute to a bin.|
|hist046_Graphics_highlight1D.C|Use the highlight mechanism to zoom on a histogram.|
|hist047_Graphics_candle_decay.C|Candle plot illustrating a time development of a certain value.|
|hist048_Graphics_candle_hist.C|Illustrate candle plot options.|
|hist049_Graphics_candle_plot.C|Create a candle plot with 2-D histograms.|
|hist050_Graphics_candle_plot_options.C|Illustrate more candle plot options.|
|hist051_Graphics_candle_plot_stack.C|Create a THStack with candle plot option.|
|hist052_Graphics_candle_plot_whiskers.C|Create a candle plot showing the whiskers definition.|
|hist053_Graphics_candle_scaled.C|Illustrate what scaling effects on candle and violin charts.|

\anchor texec

## TExec

|                                    |           **Tutorial**          |                               |     **Description**    |
|------------------------------------|---------------------------------|-------------------------------|------------------------|
|**C++ macro** |**Python** |**Python+Newest Pythonization**||
|hist057_TExec_th1.C                 |                                 |                               |Echo object at mouse position.|
|hist058_TExec_th2.C                 |                                 |                               |Echo object at mouse position and show a graphics line.|
|hist105_TExec_dynamic_slice.C       |hist105_TExec_dynamic_slice.py   ||Show the slice of a TH2 following the mouse position.|

\anchor hist_alltutorials

@}

