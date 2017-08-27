// @(#)root/mathcore:$Id$
// Author: Alejandro García Montoro 06/2017

/*************************************************************************
 * Copyright (C) 2017, CERN                                              *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMathVectorized
#define ROOT_TMathVectorized

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMathVectorized                                                      //
//                                                                      //
// Encapsulate vectorized version of the most frequently used Math      //
// functions.                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"
#include "TError.h"
#include "Math/Math_vectypes.hxx"
//
#include <algorithm>
#include <limits>
#include <cmath>

#ifdef R__HAS_VECCORE

#include <VecCore/VecCore>

/* **************************************** */
/* *         Base math functions          * */
/* * Linear versions defined in TMathBase * */
/* **************************************** */
namespace TMath {

template <class NotCompileIfScalarBackend = std::enable_if<!(std::is_same<double, ROOT::Double_v>::value)>>
inline ROOT::Double_v Min(ROOT::Double_v a, ROOT::Double_v b)
{
   return vecCore::math::Min(a, b);
}
}

/* ************************************ */
/* *       Usual math functions       * */
/* * Linear versions defined in TMath * */
/* ************************************ */
namespace TMath {

/* ************************** */
/* * Mathematical Functions * */
/* ************************** */

/* *********************** */
/* * Statistic Functions * */
/* *********************** */

template <class NotCompileIfScalarBackend = std::enable_if<!(std::is_same<double, ROOT::Double_v>::value)>>
ROOT::Double_v BreitWigner(ROOT::Double_v &x, Double_t mean = 0, Double_t gamma = 1)
{
   ROOT::Double_v bw = gamma / ((x - mean) * (x - mean) + gamma * gamma / 4);
   return bw / (2 * Pi());
}

template <class NotCompileIfScalarBackend = std::enable_if<!(std::is_same<double, ROOT::Double_v>::value)>>
ROOT::Double_v CauchyDist(ROOT::Double_v &x, Double_t t = 0, Double_t s = 1)
{
   ROOT::Double_v temp = (x - t) * (x - t) / (s * s);
   ROOT::Double_v result = 1 / (s * Pi() * (1 + temp));
   return result;
}

template <class NotCompileIfScalarBackend = std::enable_if<!(std::is_same<double, ROOT::Double_v>::value)>>
ROOT::Double_v Gaus(ROOT::Double_v &x, Double_t mean = 0, Double_t sigma = 1, Bool_t norm = kFALSE)
{
   if (sigma == 0)
      return vecCore::NumericLimits<ROOT::Double_v>::Infinity();

   ROOT::Double_v arg = (x - mean) / sigma;

   // Compute the function only when the arg meets the criteria
   ROOT::Double_v res =
      vecCore::Blend<ROOT::Double_v>(vecCore::math::Abs(arg) < 39.0, vecCore::math::Exp(-0.5 * arg * arg), 0.0);

   return norm ? res / (2.50662827463100024 * sigma) : res; // sqrt(2*Pi)=2.50662827463100024
}

template <class NotCompileIfScalarBackend = std::enable_if<!(std::is_same<double, ROOT::Double_v>::value)>>
ROOT::Double_v LaplaceDist(ROOT::Double_v &x, Double_t alpha = 0, Double_t beta = 1)
{
   ROOT::Double_v result;
   result = vecCore::math::Exp(-vecCore::math::Abs((x - alpha) / beta));
   result /= (2. * beta);
   return result;
}

template <class NotCompileIfScalarBackend = std::enable_if<!(std::is_same<double, ROOT::Double_v>::value)>>
ROOT::Double_v LaplaceDistI(ROOT::Double_v &x, Double_t alpha = 0, Double_t beta = 1)
{
   ROOT::Double_v temp = 0.5 * vecCore::math::Exp(-vecCore::math::Abs((x - alpha) / beta));
   return vecCore::Blend(x <= alpha, temp, 1 - temp);
}
} // namespace TMath

#endif // ROOT_TMathVectorized

#endif // R__HAS_VECCORE
