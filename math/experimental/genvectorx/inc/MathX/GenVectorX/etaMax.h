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

#ifndef ROOT_MathX_GenVectorX_etaMax
#define ROOT_MathX_GenVectorX_etaMax 1

#include "MathX/GenVectorX/MathHeaders.h"

#include "MathX/GenVectorX/AccHeaders.h"

#include <limits>
#include <cmath>

// #if !defined(ROOT_MATH_SYCL) && !defined(ROOT_MATH_CUDA)
// typedef long double d_type;
// #else
// typedef d_type  double ;
// #endif

namespace ROOT {

namespace ROOT_MATH_ARCH {

/**
    The following function could be called to provide the maximum possible
    value of pseudorapidity for a non-zero rho.  This is log ( max/min )
    where max and min are the extrema of positive values for type
    long double.
 */
__roohost__ __roodevice__ inline long double etaMax_impl()
{
   using std::log;
   return math_log(std::numeric_limits<long double>::max() / 256.0l) -
          math_log(std::numeric_limits<long double>::denorm_min() * 256.0l) + 16.0 * math_log(2.0);
   // Actual usage of etaMax() simply returns the number 22756, which is
   // the answer this would supply, rounded to a higher integer.
}

/**
    Function providing the maximum possible value of pseudorapidity for
    a non-zero rho, in the Scalar type with the largest dynamic range.
 */
template <class T>
inline T etaMax()
{
   return static_cast<T>(22756.0);
}

} // namespace ROOT_MATH_ARCH

} // namespace ROOT

#endif /* ROOT_MathX_GenVectorX_etaMax  */
