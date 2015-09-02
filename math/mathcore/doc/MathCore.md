/**

\defgroup MathCore  MathCore

The Core Mathematical Library of %ROOT. See the \ref MathCorePage description page.

\ingroup Math


\page MathCorePage MathCore Library

**MathCore** provides a collection of functions and C++ classes for HEP numerical computing. This library provides only the basic mathematical functions and algorithms and not all the functionality required by the HEP community. More advanced mathematical functionalities is provided by the \ref MathMore. The current set includes classes and functions for:

*   \ref SpecFunc "Basic special functions" used in HEP like the gamma, beta and error function;
*   \ref StatFunc : mathematical functions used in statistics, such as the probability density functions and the cumulative distributions functions (lower and upper integral of the pdf's) for continuous and discrete distributions.;
*   \ref CppFunctions :
    *   \ref GenFunc, including helper class to wrap free (static) and non-static member functions
    *   \ref ParamFunc
*   Numerical algorithms: user classes with (in some cases) basic implementations for:
    *   \ref Integration
    *   \ref Deriv
    *   \ref RootFinders
    *   \ref Min1D and \ref MultiMin
*   \ref Fit :classes for fitting and parameter estimation from a given data set.

Note that in this latest release the \ref Vector "GenVector" (physics and geometry vectors for 2,3 and 4 dimensions with their transformations) is not anymore part of MathCore, but is built as a separate library.   
MathCore contains instead now classes which were originally part of _libCore_. These include:

*   [**<tt>TMath</tt>**](http://root.cern.ch/root/htmldoc/math/mathcore/TMath.html): namespace with mathematical functions and basic function algorithms.
*   [**<tt>TComplex</tt>**](http://root.cern.ch/root/htmldoc/TComplex.html): class for complex numbers.
*   Random classes: base class [**<tt>TRandom</tt>**](http://root.cern.ch/root/htmldoc/TRandom.html) and the derived classes [<tt>TRandom1</tt>](http://root.cern.ch/root/htmldoc/TRandom1.html), [<tt>TRandom2</tt>](http://root.cern.ch/root/htmldoc/TRandom2.html) and [<tt>TRandom3</tt>](http://root.cern.ch/root/htmldoc/TRandom.html)), implementing the pseudo-random number generators.

MathCore and its CINT dictionary is normally built by default in %ROOT. Alternatively MathCore can be built as a stand-alone library (excluding classes like <tt>TComplex</tt> or <tt>TRandom</tt>
having a direct dependency of %ROOT _libCore_), downloading the current version from [here](../MathCore.tar.gz). Note, that the stand-alone library, in contrast to the one distributed by %ROOT, **does
not** contain the dictionary information and therefore cannot be used interactively. To build the stand-alone MathCore library run first the _configure_ script and then _make_. Do _configure --help_
to see options available in configuring, like defining the installation directory. Run also _make install_ to install the library and include files and _make check_ to build the tests.

*/
