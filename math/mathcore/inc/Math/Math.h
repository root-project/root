// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 14 15:44:38 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// mathematical constants like Pi

#ifndef ROOT_Math_Math
#define ROOT_Math_Math

#ifdef _WIN32
#define _USE_MATH_DEFINES
#define HAVE_NO_LOG1P
#define HAVE_NO_EXPM1
#endif

#include <cmath>

#if defined(__sun)
//solaris definition of cmath does not include math.h which has the definitions of numerical constants
#include <math.h>
#endif


#ifdef HAVE_NO_EXPM1
// needed to implement expm1
#include <limits>
#endif


#ifndef M_PI

#define M_PI       3.14159265358979323846264338328      // Pi
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923132169164      // Pi/2
#endif

#ifndef M_PI_4
#define M_PI_4     0.78539816339744830961566084582      // Pi/4
#endif

namespace ROOT {

   namespace Math {

/**
    Mathematical constants
*/
inline double Pi() { return M_PI; }

/**
    declarations for functions which are not implemented by some compilers
*/

/// log(1+x) with error cancelatio when x is small
inline double log1p( double x) {
#ifndef HAVE_NO_LOG1P
   return ::log1p(x);
#else
   // if log1p is not in c math library
  volatile double y;
  y = 1 + x;
  return std::log(y) - ((y-1)-x)/y ;  /* cancels errors with IEEE arithmetic */
#endif
}
/// exp(x) -1 with error cancellation when x is small
inline double expm1( double x) {
#ifndef HAVE_NO_EXPM1
   return ::expm1(x);
#else
   // compute using taylor expansion until difference is less than epsilon
   // use for values smaller than 0.5 (for larger (exp(x)-1 is fine
   if (std::abs(x) < 0.5)
   {
       // taylor series S = x + (1/2!) x^2 + (1/3!) x^3 + ...

      double i = 1.0;
      double sum = x;
      double term = x / 1.0;
      do {
         i++ ;
         term *= x/i;
         sum += term;
      }
      while (std::abs(term) > std::abs(sum) * std::numeric_limits<double>::epsilon() ) ;

      return sum ;
   }
   else
   {
      return std::exp(x) - 1;
   }
#endif
}


   } // end namespace Math

} // end namespace ROOT


// for Doxygen documentation


/**
@defgroup Math  Mathematical Libraries 


The ROOT Math package consists of the following components:

## MathCore
  
  A self-consistent minimal set of tools required for the basic numerical computing.
  It provides the major mathematical functions in the namespaces ROOT::Math and TMath,
  classes for random number generators, TRandom, class for complex numbers, TComplex,
  common interfaces for function evaluation and numerical algorithms. 
  Basic implementations of some of the numerical algorithms such as integration or derivation, are as also provided by MathCore. 
  Furthermore, classes required for fit the ROOT data objects (and also any data sets) are as well provided.

## MathMore 
 
Package incorporating advanced numerical functionality and dependent on external libraries like the GNU Scientific Library ([GSL](http://www.gnu.org/software/gsl/)). It complements the MathCore library by providing a more complete sets of special mathematical functions and implementations of the numerical algorithms interfaces defined in MathCore using GSL.

## Minimization and Fitting Libraries: 

Libraries required for numerical minimization and fitting. The minimization libraries include the numerical methods for solving the fitting problem by finding minimum of multi-dimensional
  function. The current common interface for minimization is the class ROOT::Math::Minimizer and implemented by derived classes in the minimization and fitting libraries. The fitting in ROOT is
  organized in fitting classes present in MathCore in the (ROOT::Fit namespace) for providing the fitting functionality and the use the minimization libraries via the common interface (ROOT::Math::Minimizer). In detail the minimization libraries, implementing all the new and old minimization interface, include:

- Minuit: library providing via a class TMinuit an implementation of the popular MINUIT minimization package. In addition the library contains also an implementation of the linear fitter (class TLinearFitter), for solving linear least square fits.
- Minuit2: new object-oriented implementation of MINUIT, with the same minimization algorithms (such as Migrad or Simplex). In addition it provides a new implementation of the Fumili algorithm, a specialized method for finding the minimum of a standard least square or likelihood functions. 
- Fumili: library providing the implementation of the original Fumili fitting algorithm (class TFumili).

## Linear algebra 

Two libraries are contained in ROOT for describing linear algebra matrices and vector classes:
   
- Matrix: general matrix package providing matrix classes (TMatrixD and TMatrixF)  and vector classes (TVectorD and TVectorF) and the complete environment to perform linear algebra calculations, like equation solving and eigenvalue decompositions.
- SMatrix: package optimized for high performances matrix and vector computations of small and fixed size. It is based on expression templates to achieve an high level optimization.


## Physics Vectors

Classes for describing vectors in 2, 3 and 4 dimensions (relativistic vectors) and their rotation and transformation algorithms. Two package exist in ROOT:

- Physics: library with the TVector3 and TLorentzVector classes.
- GenVector: new library providing generic class templates for modeling the vectors.

## Unuran

Package with universal algorithms for generating non-uniform pseudo-random numbers, from a large classes of continuous or discrete distributions in one or multi-dimensions. 

## Foam 

Multi-dimensional general purpose Monte Carlo event generator (and integrator). It generates randomly points (vectors) according to an arbitrary probability distribution  in n dimensions.

## FFTW

Library with implementation of the fast Fourier transform (FFT) using the FFTW package. It requires a previous installation of [FFTW](http://www.fftw.org)>FFTW).

## MLP

Library with the neural network class, TMultiLayerPerceptron based on the NN algorithm from the mlpfit package.

## Quadp

Optimization library with linear and quadratic programming methods. It is based on the Matrix package.


Further information is available at the following links:

- The Math chapter in the user guide
- The Linear Algebra chapter in the user guide
- The Physics Vector chapter in the user guide
- [Inventory of Math functions and algorithms] (http://project-mathlibs.web.cern.ch/project-mathlibs/mathTable.html)

**/

/**
@defgroup MathCore  Mathematical Libraries 

\htmlinclude ../../math/mathcore/doc/index.html

@ingroup Math

**/


#endif /* ROOT_Math_Math */
