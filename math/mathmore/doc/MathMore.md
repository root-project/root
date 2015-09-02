
/** 

  \defgroup MathMore  MathMore

    The Mathematical library providing some advanced functionality and based on GSL. See the \ref MathMorePage page. 

    \ingroup Math



\page MathMorePage MathMore Library

**MathMore** provides an advanced collection of functions and C++ classes for HEP numerical computing. This is an extension of the functionality provided by the \ref MathCore. The current set includes classes and functions for:

*   \ref SpecFunc, with all the major functions (Bessel functions, Legendre polynomial, etc..)
*   \ref StatFunc, Mathematical functions used in statistics such as probability density functions, cumulative distributions functions and their inverse (quantiles).
*   Numerical algorithms:
    *   \ref Integration
    *   \ref MCIntegration
    *   \ref Deriv
    *   \ref RootFinders
    *   \ref Min1D
    *   \ref MultiMin
*   \ref Interpolation
*   \ref FuncApprox, based on Chebyshev polynomials
*   \ref Random

The mathematical functions are implemented as a set of free functions in the namespace \em ROOT::Math. The naming used for the special functions is the same proposed for the C++ standard (see C++ standard extension [proposal document](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1687.pdf)). The MathMore library is implemented wrapping in C++ the GNU Scientific Library ([GSL](http://www.gnu.org/software/gsl)).   
To build MathMore you need to have first GSL installed somewhere in your system. A version of GSL larger or equal 1.8 is required. A tar file of GSL can be downloaded from the [GSL Web site](http://www.gnu.org/software/gsl/#downloading), or (for version 1.8) from [here](http://seal.web.cern.ch/seal/MathLibs/gsl-1.8.tar.gz). Windows binaries, compiled using Visual Studio 7.1 can be downloaded from [this location](http://seal.web.cern.ch/seal/MathLibs/GSL-1.8.zip).

MathMore (and its %ROOT CINT dictionary) can be built within %ROOT whenever a GSL library is found in the system. Optionally the GSL library and header file location can be specified in the %ROOT configure script with _configure --with-gsl-incdir=... --with-gsl-libdir=..._

MathMore can be built also a stand-alone library (without requiring %ROOT) downloding the tar file from the Web. Note that in this case the library will **NOT** contain the dictionary information and therefore cannot be used interactivly.   
The current standalone version of MathMore can be downloaded [here](../MathMore.tar.gz).   
To build the library run first the script _configure --with-gsl="gsl_dir_location"_ and then _make_. Do _configure --help_ to see options available in configuring, like defining the installation directory. Run also _make install_ to install the library and include files and _make check_ to build the tests.   
The source code is distributed under the GNU General Public License.

*/
