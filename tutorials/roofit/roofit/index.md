\addtogroup tutorial_roofit_main RooFit
\ingroup tutorial_roofit

@{
Explore the tutorials below to discover the main features of RooFit. A more indepth description of the codes can be found at [RooFit User Manual](https://root.cern/download/doc/RooFit_Users_Manual_2.91-33.pdf)


## Tutorials sorted after groups

- [Basic Functionalities](\ref roofit_basic)
- [Addition and Convolution](\ref roofit_addition_convolution)
- [Multidimensional Models](\ref roofit_multidimentional_models)
- [Data and Categories](\ref roofit_data_categories)
- [Organisation and Simultaneous Fits](\ref roofit_organisation_fits)
- [Likelihood and Minimization](\ref roofit_likelihood_maximization)
- [Special Pdf's](\ref roofit_special_pdfs)
- [Validation and MC Studies](\ref roofit_validation_mc)
- [Numeric Algorithm Tuning](\ref roofit_numeric_algorithm_tuning)
- [Misc.](\ref roofit_misc)


[List of all tutorials](\ref roofit_alltutorials)

\anchor roofit_basic
## Introduction

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf101_basics.C| rf101_basics.py | Fitting, plotting, toy data generation on one-dimensional PDFs.|
| rf102_dataimport.C | rf102_dataimport.py | Importing data from ROOT TTrees and THx histograms.|
| rf103_interprfuncs.C | rf103_interprfuncs.py | Interpreted functions and PDFs.|
| rf104_classfactory.C | rf104_classfactory.py | The class factory for functions and pdfs.|
| rf105_funcbinding.C | rf105_funcbinding.py | Binding ROOT math functions as RooFit functions and pdfs.|
| rf106_plotdecoration.C | rf106_plotdecoration.py | Adding boxes with parameters, statistics to RooPlots, decorating with arrows, text etc...|
| rf107_plotstyles.C | rf107_plotstyles.py | Various plotting styles of data, functions in a RooPlot.|
| rf108_plotbinning.C | rf108_plotbinning.py | Plotting unbinned data with alternate and variable binnings.|
| rf109_chi2residpull.C | rf109_chi2residpull.py | Calculating chi^2 from histograms and curves in RooPlots, making histogram of residual and pull distributions.|
| rf110_normintegration.C | rf110_normintegration.py | Normalization and integration of pdfs, construction of cumulative distribution monodimensional functions.|
| rf111_derivatives.C | rf111_derivatives.py | Numerical 1st,2nd and 3rd order derivatives w.r.t. observables and parameters.|


\anchor roofit_addition_convolution
## Addition and Convolution

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf201_composite.C| rf201_composite.py | Composite pdf with signal and background component.|
| rf202_extendedmlfit.C| rf202_extendedmlfit.py | Setting up an extended maximum likelihood fit.|
| rf203_ranges.C| rf203_ranges.py | Fitting and plotting in sub ranges.|
| rf205_compplot.C| rf205_compplot.py | Options for plotting components of composite pdfs.|
| rf206_treevistools.C| rf206_treevistools.py | Tools for visualization of RooAbsArg expression trees.|
| rf207_comptools.C| rf207_comptools.py | Tools and utilities for manipulation of composite objects.|
| rf208_convolution.C| rf208_convolution.py | One-dimensional numeric convolution.|
| rf209_anaconv.C| rf209_anaconv.py |decay function pdfs with optional B physics effects (mixing and CP violation).|
| rf210_angularconv.C| rf210_angularconv.py | Convolution in cyclical angular observables theta.|
| rf211_paramconv.C| rf211_paramconv.py | Working with a pdf with a     convolution operator in terms of a parameter.|

\anchor roofit_multidimentional_models
## Multidimensional Models

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf301_composition.C| rf301_composition.py | Multi-dimensional pdfs through composition, e.g. substituting a pdf parameter with a function that depends on other observables.|
| rf302_utilfuncs.C| rf302_utilfuncs.py | Utility functions classes available for use in tailoring of composite (multidimensional) pdfs.|
| rf303_conditional.C| rf303_conditional.py | Use of tailored pdf as conditional pdfs.s.|
| rf304_uncorrprod.C| rf304_uncorrprod.py | Simple uncorrelated multi-dimensional pdfs.|
| rf305_condcorrprod.C| rf305_condcorrprod.py | Multi-dimensional pdfs with conditional pdfs in product.|
| rf306_condpereventerrors.C| rf306_condpereventerrors.py | Conditional pdf with per-event errors.|
| rf307_fullpereventerrors.C| rf307_fullpereventerrors.py | Full pdf with per-event errors.|
| rf308_normintegration2d.C| rf308_normintegration2d.py | Normalization and integration of pdfs, construction of cumulative distribution functions from pdfs in two dimensions.|
| rf309_ndimplot.C| rf309_ndimplot.py | Making 2/3 dimensional plots of pdfs and datasets.|
| rf310_sliceplot.C| rf310_sliceplot.py | Projecting pdf and data slices in discrete observables.|
| rf311_rangeplot.C| rf311_rangeplot.py | Projecting pdf and data ranges in continuous observables.|
| rf312_multirangefit.C| rf312_multirangefit.py | Performing fits in multiple (disjoint) ranges in one or more dimensions.|
| rf313_paramranges.C| rf313_paramranges.py | Working with parametrized ranges to define non-rectangular regions for fitting and integration.|
| rf314_paramfitrange.C| rf314_paramfitrange.py | Working with parametrized ranges in a fit. This an example of a fit with an acceptance that changes per-event.|
| rf315_projectpdf.C| rf315_projectpdf.py | Marginizalization of multi-dimensional pdfs through integration.|
| rf316_llratioplot.C| rf316_llratioplot.py | Using the likelihood ratio technique to construct a signal enhanced one-dimensional projection of a multi-dimensional pdf.|


\anchor roofit_data_categories
## Data and Categories

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf401_importttreethx.C| rf401_importttreethx.py | Advanced options for importing data from ROOT TTree and THx histograms.|
| rf402_datahandling.C| rf402_datahandling.py | Tools for manipulation of (un)binned datasets.|
| rf403_weightedevts.C| rf403_weightedevts.py | Using weights in unbinned datasets.|
| rf404_categories.C| rf404_categories.py | Working with RooCategory objects to describe discrete variables.|
| rf405_realtocatfuncs.C| rf405_realtocatfuncs.py | Demonstration of real-->discrete mapping functions.|
| rf406_cattocatfuncs.C| rf406_cattocatfuncs.py | Demonstration of discrete-->discrete (invertible) functions.|
| rf407_ComputationalGraphVisualization.C| rf407_ComputationalGraphVisualization.py | Visualing computational graph model before fitting, and latex printing of lists and sets of RooArgSets after fitting.|

\anchor roofit_organisation_fits
## Organisation and Simultaneous Fits

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf501_simultaneouspdf.C| rf501_simultaneouspdf.py | Using simultaneous pdfs to describe simultaneous fits to multiple datasets.|
| rf502_wspacewrite.C| rf502_wspacewrite.py | Creating and writing a workspace.|
| rf503_wspaceread.C| rf503_wspaceread.py | Reading and using a workspace.|
| rf504_simwstool.C| rf504_simwstool.py | Using RooSimWSTool to construct a simultaneous pdf that is built of variations of an input pdf.|
| rf505_asciicfg.C| rf505_asciicfg.py | Reading and writing ASCII configuration files.|
| rf506_msgservice.C| rf506_msgservice.py | Tuning and customizing the RooFit message logging facility.|
| rf508_listsetmanip.C| rf508_listsetmanip.py | RooArgSet and RooArgList tools and tricks.|
| rf510_wsnamedsets.C| rf510_wsnamedsets.py | Working with named parameter sets and parameter snapshots in workspaces.|
| rf511_wsfactory_basic.C| rf511_wsfactory_basic.py | Basic use of the 'object factory' associated with a workspace to rapidly build pdfs functions and their parameter components.|
| rf512_wsfactory_oper.C| rf512_wsfactory_oper.py | Pperator expressions and expression-based basic pdfs in the workspace factory syntax.|
| rf513_wsfactory_tools.C| rf513_wsfactory_tools.py | RooCustomizer and RooSimWSTool interface in factory workspace tool in a complex standalone B physics example.|

\anchor roofit_likelihood_maximization
## Likelihood and Minimization

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf601_intminuit.C| rf601_intminuit.py | Interactive minimization with MINUIT.|
| rf602_chi2fit.C| rf602_chi2fit.py | Setting up a chi^2 fit to a binned dataset.|
| rf604_constraints.C| rf604_constraints.py | Fitting with constraints.|
| rf605_profilell.C| rf605_profilell.py | Working with the profile likelihood estimator.|
| rf606_nllerrorhandling.C| rf606_nllerrorhandling.py | Understanding and customizing error handling in likelihood evaluations.|
| rf607_fitresult.C| rf607_fitresult.py | Demonstration of options of the RooFitResult class.|
| rf608_fitresultaspdf.C| rf608_fitresultaspdf.py | Representing the parabolic approximation of the fit as a multi-variate Gaussian on the parameters of the fitted pdf.|
| rf609_xychi2fit.C| rf609_xychi2fit.py | Setting up a chi^2 fit to an unbinned dataset with X,Y,err(Y) values (and optionally err(X) values).|
| rf610_visualerror.C| rf610_visualerror.py | Visualization of errors from a covariance matrix.|
| rf611_weightedfits.C| | Parameter uncertainties for weighted unbinned ML fits.|
| rf612_recoverFromInvalidParameters.C| rf612_recoverFromInvalidParameters.py | Recover from regions where the function is not defined.|

\anchor roofit_special_pdfs
## Special Pdf's

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf701_efficiencyfit.C| rf701_efficiencyfit.py | Unbinned maximum likelihood fit of an efficiency eff(x) function.|
| rf702_efficiencyfit_2D.C| rf702_efficiencyfit_2D.py | Unbinned maximum likelihood fit of an efficiency eff(x) function to a dataset D(x,cut), cut is a category encoding a selection whose efficiency as function of x should be described by eff(x).|
| rf703_effpdfprod.C| rf703_effpdfprod.py | Using a product of an (acceptance) efficiency and a pdf as pdf.|
| rf704_amplitudefit.C| rf704_amplitudefit.py | Using a pdf defined by a sum of real-valued amplitude components.|
| rf705_linearmorph.C| rf705_linearmorph.py | Linear interpolation between pdf shapes using the 'Alex Read' algorithm.|
| rf706_histpdf.C| rf706_histpdf.py | Histogram-based pdfs and functions.|
| rf707_kernelestimation.C| rf707_kernelestimation.py | Using non-parametric (multi-dimensional) kernel estimation pdfs.|
| rf708_bphysics.C| rf708_bphysics.py | Special decay pdf for B physics with mixing and/or CP violation.|

\anchor roofit_validation_mc
## Validation and MC Studies

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf801_mcstudy.C| rf801_mcstudy.py | Toy Monte Carlo study that perform cycles of event generation and fitting.|
| rf802_mcstudy_addons.C| | RooMCStudy - using separate fit and generator models, using the chi^2 calculator model. Running a biased fit model against an optimal fit.|
| rf803_mcstudy_addons2.C| | RooMCStudy - Using the randomizer and profile likelihood add-on models.|
| rf804_mcstudy_constr.C| | Using RooMCStudy on models with constrains.|


\anchor roofit_numeric_algorithm_tuning
## Numeric Algorithm Tuning

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf901_numintconfig.C| rf901_numintconfig.py | Configuration and customization of how numeric (partial) integrals are executed.|
| rf902_numgenconfig.C| rf902_numgenconfig.py | Configuration and customization of how MC sampling algorithms on specific pdfs are executed.|
| rf903_numintcache.C| rf903_numintcache.py | Caching of slow numeric integrals and parameterization of slow numeric integrals.|


\anchor roofit_misc
## Misc.

| **Tutorial** || **Description** |
|------|--------|-----------------|
| rf204a_extendedLikelihood.C| rf204a_extendedLikelihood.py | Extended maximum likelihood fit in multiple ranges.|
| rf204b_extendedLikelihood_rangedFit.C| rf204b_extendedLikelihood_rangedFit.py | This macro demonstrates how to set up a fit in two ranges for plain likelihoods and extended likelihoods.|
| rf212_plottingInRanges_blinding.C| rf212_plottingInRanges_blinding.py | Plot a PDF in disjunct ranges, and get normalisation right.|
| rf408_RDataFrameToRooFit.C| rf408_RDataFrameToRooFit.py | Fill RooDataSet/RooDataHist in RDataFrame.|
| | rf409_NumPyPandasToRooFit.py | Convert between NumPy arrays or Pandas DataFrames and RooDataSets.|
| rf514_RooCustomizer.C| rf514_RooCustomizer.py | Using the RooCustomizer to create multiple PDFs that share a lot of properties, but have unique parameters for each category. As an extra complication, some of the new parameters need to be functions of a mass parameter.|
| | rf515_hfJSON.py | With the HS3 standard, it is possible to code RooFit-Models of any kind as JSON files. In this tutorial, you can see how to code up a (simple) HistFactory-based model in JSON and import it into a RooWorkspace.|
| rf613_global_observables.C| rf613_global_observables.py | This tutorial explains the concept of global observables in RooFit, and showcases how their values can be stored either in the model or in the dataset.|
| rf614_binned_fit_problems.C| rf614_binned_fit_problems.py | A tutorial that explains you how to solve problems with binning effects and numerical stability in binned fits.|
| | rf615_simulation_based_inference.py | Use Simulation Based Inference (SBI) in RooFit.|
| rf616_morphing.C| rf616_morphing.py | Use Morphing in RooFit.|
| | rf617_simulation_based_inference_multidimensional.py | Use Simulation Based Inference (SBI) in multiple dimensions in RooFit.|
| | rf618_mixture_models.py | Use of mixture models in RooFit.|
| rf709_BarlowBeeston.C| rf709_BarlowBeeston.py | Implementing the Barlow-Beeston method for taking into account the statistical uncertainty of a Monte-Carlo fit template.|
| rf710_roopoly.C| rf710_roopoly.py | Taylor expansion of RooFit functions using the taylorExpand function with RooPolyFunc.|
| rf711_lagrangianmorph.C| rf711_lagrangianmorph.py | Morphing effective field theory distributions with RooLagrangianMorphFunc. A morphing function as a function of one coefficient is setup and can be used to obtain the distribution for any value of the coefficient.|
| rf712_lagrangianmorphfit.C| rf712_lagrangianmorphfit.py | Performing a simple fit with RooLagrangianMorphFunc. A morphing function is setup as a function of three variables and a fit is performed on a pseudo-dataset.|

\anchor roofit_alltutorials

@}
