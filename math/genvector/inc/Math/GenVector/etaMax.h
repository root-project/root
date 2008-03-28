// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , FNAL MathLib Team                             *
  *                                                                    *
  *                                                                    *
  **********************************************************************/


// Header source file for function etaMax
//
// Created by: Mark Fischler  at Thu Jun 2 2005


#ifndef ROOT_Math_GenVector_etaMax 
#define ROOT_Math_GenVector_etaMax  1


#include <limits>
#include <cmath>


namespace ROOT {

  namespace Math {

    /**
        The following function could be called to provide the maximum possible
        value of pseudorapidity for a non-zero rho.  This is log ( max/min )
        where max and min are the extrema of positive values for type
        long double.
     */
    inline
    long double etaMax_impl() {
      return std::log ( std::numeric_limits<long double>::max()/256.0l ) -
             std::log ( std::numeric_limits<long double>::denorm_min()*256.0l )
             + 16.0 * std::log(2.0);
    // Actual usage of etaMax() simply returns the number 22756, which is
    // the answer this would supply, rounded to a higher integer.
    }

    /**
        Function providing the maximum possible value of pseudorapidity for
        a non-zero rho, in the Scalar type with the largest dynamic range.
     */
    template <class T>
    inline
    T etaMax() {
      return static_cast<T>(22756.0);
    }

  } // namespace Math

} // namespace ROOT


#endif /* ROOT_Math_GenVector_etaMax  */
