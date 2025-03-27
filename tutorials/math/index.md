\addtogroup tutorial_math

@{
## Table of contents
- [Basic features of MathCore](\ref mathcore)
- [Basic features of MathMore (GSL binding)](\ref mathmore)
- [Fast Fourier Transform](\ref tutorial_fft)
- [Fitting](\ref tutorial_fit)
- [Histogramming related features](\ref histograms)
- [Linear algebra](\ref tutorial_matrix)
- [Minimizer features](\ref minimization)
- [Physics vectors](\ref genvector)
- [R binding](\ref tutorial_r)
- [Random number generation with Unuran](\ref tutorial_unuran)
- [RVec basic features](\ref tutorial_vecops)


\anchor mathcore
## Basic features of MathCore

To get started these examples show some of the basic features of the MathCore library.

| **Tutorial** || **Description** |
|------|--------|-----------------|
| ChebyshevPol.C |  | Example of Chebyshev polynomials using TFormula pre-defined definitions of chebyshev polynomials. |
|  | exampleFunction.py | Example of using Python functions as inputs to numerical algorithms using the ROOT Functor class. |
| exampleFunctor.C |  | Tutorial illustrating how to create a TF1 class using c++ functors or class member functions. |
| GammaFun.C |  | Example showing the usage of the major special math functions (gamma, beta, erf) in ROOT.|
| goftest.C |  | Example showing usage of goodness of fit tests.|
| kdTreeBinning.C |  | Example binning the data in cells of equal content using a kd-tree.|
| limit.C |  | This example shows random number generation for filling histograms.|
| mathcoreSpecFunc.C |  | Example macro showcasing some special mathematical functions.|
| multidimSampling.C |  | Example of random number generation by sampling a multi-dim distribution using the DistSampler class.|
| permute.C |  | Tutorial illustrating the use of TMath::Permute for computing all permutations of n natural numbers.|
| testrandom.C |  | Performance test of all the ROOT random generator (TRandom, TRandom1, TRandom2 and TRandom3).|

\anchor mathmore
## Basic features of MathMore

To get started these examples show some of the basic features of the MathMore library, a package incorporating advanced numerical functionality and dependent on external libraries like the GNU Scientific Library (GSL).

| **Tutorial** || **Description** |
|------|--------|-----------------|
| exampleMultiRoot.C | | Example of using multiroot finder based on GSL algorithm. |
| Legendre.C | Legendre.py | Example of first few Legendre Polynomials. |
| LegendreAssoc.C| | Example describing the usage of different kinds of Associate Legendre Polynomials. |
| mathmoreIntegration.C |  | Example on the  usage of the adaptive 1D integration algorithm of MathMore.|
| mathmoreIntegrationMultidim.C |  | Example on the usage of the multidimensional integration algorithm of MathMore.|
| quasirandom.C |  | Example of quasi-random numbers generation.|

\anchor genvector
## Physics vectors

Examples showing usage of the GenVector library, generic class templates for modeling vectors in 2, 3 and 4 dimensions (relativistic vectors) and their rotation and transformation algorithms. 

| **Tutorial** | **Description** |
|--------------|-----------------|
| mathcoreGenVector.C | Example macro testing available methods and operation of the GenVector classes. |
| mathcoreVectorCollection.C | Example showing how to write and read a std vector of ROOT::Math LorentzVector in a ROOT tree. |
| mathcoreVectorFloatIO.C | Macro illustrating  I/O with Lorentz Vectors of floats. |
| mathcoreVectorIO.C | Example of  I/O of a GenVector Lorentz Vectors in a Tree and comparison with legacy TLorentzVector.|


\anchor histograms
## Histogramming related features

Examples showing usage of mathematical features for histigrams.

| **Tutorial** || **Description** |
|------|--------|-----------------|
| chi2test.C |  | Example to use chi2 test for comparing two histograms. |
| exampleTKDE.C |  | Example of using the TKDE class (kernel density estimator). |
| hlquantiles.C |  | Demo for quantiles (with highlight mode). |
| principal.C | principal.py | Principal Components Analysis (PCA) example. |
| qa2.C |  | Test generation of random numbers distributed according to a function defined by the user. |
| quantiles.C |  | Demo for quantiles. |

\anchor minimization
## Minimizer features

Examples showing usage of ROOT minimizers.

| **Tutorial** | **Description** |
|--------------|-----------------|
| NumericalMinimization.C |  Example on how to use the Minimizer class in ROOT. |
@}

\defgroup tutorial_fft Fast Fourier Transforms tutorials
\ingroup tutorial_math
\brief Example showing the Fast Fourier Transforms interface in ROOT.

\defgroup tutorial_fit Fit Tutorials
\ingroup tutorial_math
\brief These tutorials illustrate the main fitting features. Their names are related to the aspect which is treated in the code.

\defgroup tutorial_FOAM FOAM tutorials
\ingroup tutorial_math
\brief Examples showing how to use FOAM.

\defgroup tutorial_matrix Matrix tutorials
\ingroup tutorial_math
\brief Examples showing how to use TMatrix.

\defgroup tutorial_pdf Probability distributions tutorials
\ingroup tutorial_math
\brief Examples showing the available probability distributions in ROOT.

\defgroup tutorial_quadp Quadratic programming package
\ingroup tutorial_math
\brief Example showing the usage of the quadratic programming package quadp.

\defgroup tutorial_r R tutorials
\ingroup tutorial_math
\brief Examples showing the R interface.

\defgroup tutorial_unuran Unuran tutorials
\ingroup tutorial_math
\brief Examples showing unuran capabilities.

\defgroup tutorial_vecops VecOps tutorials
\ingroup tutorial_math
\brief These examples show the functionalities of the VecOps utilities.