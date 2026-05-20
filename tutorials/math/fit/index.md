\addtogroup tutorial_fit
 
In the tables below the tutorials are grouped according to what is fitted, e.g histograms and graphs, and the dimensionality. In addition, the tables include a column with the fit funcions used in the tutorials (e.g gaus, expo or user defined functions), and how they are combined (e.g sum or convolution).
   
Explore the examples below or go to the [Fitting user guide](https://root.cern/manual/fitting/). A list of all the Fit tutorials can be found [here](\ref fit_alltutorials).

## Table of contents
- [Histograms 1D](\ref histo_1D)
- [Histograms 2D](\ref histo_2D)
- [Graphs 1D](\ref graphs_1D)
- [Graphs 2D](\ref graphs_2D)
- [Higher dimensional histograms and graphs](\ref higher_dim)


\anchor histo_1D
## Histograms 1D

|          **Tutorial**          || **Description** | **Fit functions** |
|---------------|-----------------|-----------------|--------------------|
| fit1.C | | Simple fitting example (1-d histogram with an interpreted function). | user defined |
| | fit1.py | Fit example. | |
| myfit.C | | Get in memory an histogram from a root file and fit a user defined function. | user defined |
| vectorizedFit.C | | Tutorial for creating a Vectorized TF1 function using a formula expression and use it for fitting an histogram. | gaus |
| ErrorIntegral.C | | Estimate the error in the integral of a fitted function taking into account the errors in the parameters resulting from the fit. | user defined |
| TestBinomial.C | | Perform a fit to a set of data with binomial errors like those derived from the division of two histograms. | user defined |
| ConfidenceIntervals.C | | Illustrates TVirtualFitter::GetConfidenceIntervals. This method computes confidence intervals for the fitted function. | linear, gaus, user defined |
| minuit2GausFit.C | | Perform fits with different configurations using Minuit2. | gaus |
| fitcont.C | | Example illustrating how to draw the n-sigma contour of a Minuit fit. | gaus |
| combinedFit.C | combinedFit.py | Combined (simultaneous) fit of two histogram with separate functions and some common parameters. | expo + gaus|
| fitNormSum.C | fitNormSum.py | Tutorial for normalized sum of two functions Here: a background exponential and a crystalball function. | expo + crystalball |
| FittingDemo.C | | Example for fitting signal/background. | quadratic + Lorenzian |
| fithist.C | | Example of fit where the model is histogram + function. | histogram + function |
| fitExclude.C | | Illustrates how to fit excluding points in a given range. | multiple linear |
| multifit.C | multifit.py | Fitting multiple functions to different ranges of a 1-D histogram. | multiple gaus |
| fitConvolution.C | fitConvolution.py | Tutorial for convolution of two functions. | expo * gaus |
| langaus.C | | Convoluted Landau and Gaussian Fitting Function (using ROOT's Landau and Gauss functions). | Landau * gaus |
| minuit2FitBench.C | | Demonstrate performance and usage of Minuit2 and Fumili2 for monodimensional fits. | Landau * gaus |
| fitFraction.C | | Fit accounting for data and Monte Carlo statistical uncertainties. | cos, user defined |


\anchor histo_2D
## Histograms 2D

| **Tutorial**  | **Description** | **Fit functions** |
|---------------|-----------------|--------------------|
| fit2.C | Fitting a 2-D histogram. | user defined |
| fit2a.C | Fitting a 2-D histogram (a variant). | user defined |
| fit2d.C | Example illustrating how to fit a 2-d histogram of type y=f(x). | pol1 |
| TwoHistoFit2D.C | Example to fit two histograms at the same time. | user defined |
| fit2dHist.C | Example to fit two histograms at the same time via the Fitter class. | user defined |
| fitslicesy.C | Illustrates how to use the TH1::FitSlicesY function. It uses the TH2F histogram generated in macro hsimple.C | |
| minuit2FitBench2D.C | Minuit2 fit 2D benchmark. | gaus |



\anchor graphs_1D
## Graphs 1D

| **Tutorial**  | **Description** |  **Fit functions**  |
|---------------|-----------------|--------------------|
| fitLinear.C | Example of fitting with a linear function, using TLinearFitter. This example is for a TGraphErrors, but it can also be used when fitting a histogram, a TGraph2D or a TMultiGraph. | user defined |
| fitLinearRobust.C | This tutorial shows how the least trimmed squares regression, included in the TLinearFitter class, can be used for fitting in cases when the data contains outliers. | qubic |
| fitMultiGraph.C | Fitting a parabola to a multigraph of 3 partly overlapping graphs with different errors. | quadratic |
| ConfidenceIntervals.C | Illustrates TVirtualFitter::GetConfidenceIntervals. This method computes confidence intervals for the fitted function. | linear, gaus, user defined |

\anchor graphs_2D
## Graphs 2D

| **Tutorial**  | **Description** |  **Fit functions**  |
|---------------|-----------------|--------------------|
| graph2dfit.C | Fitting a TGraph2D. | user defined |
| line3Dfit.C | Fitting of a TGraph2D with a 3D straight line. | linear |
| fitCircle.C | Generate points distributed with some errors around a circle. Fit a circle through the points and draw. | circle |
| ConfidenceIntervals.C | Illustrates TVirtualFitter::GetConfidenceIntervals. This method computes confidence intervals for the fitted function. | linear, gaus, user defined |

\anchor higher_dim
## Higher dimensional histograms and graphs

| **Tutorial**  | **Description** |  **Fit functions**  |
|---------------|-----------------|--------------------|
| exampleFit3D.C | Example of fitting a 3D function. Typical multidimensional parametric regression where the predictor depends on 3 variables. | user defined |
| fitLinear2.C | Fit a 5d hyperplane by n points, using the linear fitter directly. | linear |
| multidimfit.C | Multi-Dimensional Parametrisation and Fitting. | user defined |
| Ifit.C | Example of a program to fit non-equidistant data points. | user defined |

\anchor fit_alltutorials
@}
