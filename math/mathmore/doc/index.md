\defgroup MathMore  MathMore
\ingroup MathCore
\brief The Mathematical library providing GSL extensions to MathCore.

**MathMore** provides an advanced collection of functions and C++ classes for HEP numerical
computing. This is an extension of the functionality provided by the \ref MathCore, which becomes
available when a GSL library is found in the system. It can be enabled/disabled
when configuring %ROOT using `cmake -Dmathmore=On/Off`.
MathMore links with the GSL static libraries, so note that its license differs from ROOT's usual license,
since the GSL is distributed under the GNU General Public License. On some platforms (like Linux x86-64), GSL
needs to be compiled with the option `--with-pic`.

MathMore provides extensions to the following MathCore groups:
*   \ref SpecFunc, with all the major functions (Bessel functions, Legendre polynomial, etc..)
*   \ref StatFunc, Mathematical functions used in statistics such as probability density
     functions, cumulative distributions functions and their inverse (quantiles).
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

The mathematical functions are implemented as a set of free functions in the namespace \em
ROOT::Math. The naming used for the special functions is the same proposed for the C++
standard (see C++ standard extension [proposal document](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1687.pdf)).
The MathMore library is implemented wrapping in C++ the GNU Scientific Library
([GSL](http://www.gnu.org/software/gsl)). To build MathMore you need to have first GSL
installed somewhere in your system. A version of GSL larger or equal 1.8 is required. A tar
file of GSL can be downloaded from the [GSL Web site](http://www.gnu.org/software/gsl/#downloading),
or (for version 1.8) from [here](http://seal.web.cern.ch/seal/MathLibs/gsl-1.8.tar.gz).
Windows binaries, compiled using Visual Studio 7.1 can be downloaded from
[this location](http://seal.web.cern.ch/seal/MathLibs/GSL-1.8.zip).