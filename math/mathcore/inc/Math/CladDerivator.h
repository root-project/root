/// \file CladDerivator.h
///
/// \brief The file is a bridge between ROOT and clad automatic differentiation
/// plugin.
///
/// \author Vassil Vassilev <vvasilev@cern.ch>
///
/// \date July, 2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef CLAD_DERIVATOR
#define CLAD_DERIVATOR

#ifndef __CLING__
#error "This file must not be included by compiled programs."
#endif //__CLING__

#include <plugins/include/clad/Differentiator/Differentiator.h>
#include "TMath.h"
namespace clad {
namespace custom_derivatives {
namespace TMath {
template <typename T>
ValueAndPushforward<T, T> Abs_pushforward(T x, T d_x)
{
   return {::TMath::Abs(x), ((x < 0) ? -1 : 1) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> ACos_pushforward(T x, T d_x)
{
   return {::TMath::ACos(x), (-1. / ::TMath::Sqrt(1 - x * x)) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> ACosH_pushforward(T x, T d_x)
{
   return {::TMath::ACosH(x), (1. / ::TMath::Sqrt(x * x - 1)) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> ASin_pushforward(T x, T d_x)
{
   return {::TMath::ASin(x), (1. / ::TMath::Sqrt(1 - x * x)) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> ASinH_pushforward(T x, T d_x)
{
   return {::TMath::ASinH(x), (1. / ::TMath::Sqrt(x * x + 1)) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> ATan_pushforward(T x, T d_x)
{
   return {::TMath::ATan(x), (1. / (x * x + 1)) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> ATanH_pushforward(T x, T d_x)
{
   return {::TMath::ATanH(x), (1. / (1 - x * x)) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Cos_pushforward(T x, T d_x)
{
   return {::TMath::Cos(x), -::TMath::Sin(x) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> CosH_pushforward(T x, T d_x)
{
   return {::TMath::CosH(x), ::TMath::SinH(x) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Erf_pushforward(T x, T d_x)
{
   return {::TMath::Erf(x), (2 * ::TMath::Exp(-x * x) / ::TMath::Sqrt(::TMath::Pi())) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Erfc_pushforward(T x, T d_x)
{
   return {::TMath::Erfc(x), -Erf_pushforward(x, d_x).pushforward};
}

template <typename T>
ValueAndPushforward<T, T> Exp_pushforward(T x, T d_x)
{
   return {::TMath::Exp(x), ::TMath::Exp(x) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Hypot_pushforward(T x, T y, T d_x, T d_y)
{
   return {::TMath::Hypot(x, y), x / ::TMath::Hypot(x, y) * d_x + y / ::TMath::Hypot(x, y) * d_y};
}

template <typename T, typename U>
void Hypot_pullback(T x, T y, U p, clad::array_ref<T> d_x, clad::array_ref<T> d_y)
{
   T h = ::TMath::Hypot(x, y);
   *d_x += x / h * p;
   *d_y += y / h * p;
}

template <typename T>
ValueAndPushforward<T, T> Log_pushforward(T x, T d_x)
{
   return {::TMath::Log(x), (1. / x) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Log10_pushforward(T x, T d_x)
{
   return {::TMath::Log10(x), (1.0 / (x * ::TMath::Ln10())) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Log2_pushforward(T x, T d_x)
{
   return {::TMath::Log2(x), (1.0 / (x * ::TMath::Log(2.0))) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Max_pushforward(T x, T y, T d_x, T d_y)
{
   T derivative = 0;
   if (x >= y)
      derivative = d_x;
   else
      derivative = d_y;
   return {::TMath::Max(x, y), derivative};
}

template <typename T, typename U>
void Max_pullback(T a, T b, U p, clad::array_ref<T> d_a, clad::array_ref<T> d_b)
{
   if (a >= b)
      *d_a += p;
   else
      *d_b += p;
}

template <typename T>
ValueAndPushforward<T, T> Min_pushforward(T x, T y, T d_x, T d_y)
{
   T derivative = 0;
   if (x <= y)
      derivative = d_x;
   else
      derivative = d_y;
   return {::TMath::Min(x, y), derivative};
}

template <typename T, typename U>
void Min_pullback(T a, T b, U p, clad::array_ref<T> d_a, clad::array_ref<T> d_b)
{
   if (a <= b)
      *d_a += p;
   else
      *d_b += p;
}

template <typename T>
ValueAndPushforward<T, T> Power_pushforward(T x, T y, T d_x, T d_y)
{
   T pushforward = y * ::TMath::Power(x, y - 1) * d_x;
   if (d_y) {
      pushforward += (::TMath::Power(x, y) * ::TMath::Log(x)) * d_y;
   }
   return {::TMath::Power(x, y), pushforward};
}

template <typename T, typename U>
void Power_pullback(T x, T y, U p, clad::array_ref<T> d_x, clad::array_ref<T> d_y)
{
   auto t = pow_pushforward(x, y, 1, 0);
   *d_x += t.pushforward * p;
   t = pow_pushforward(x, y, 0, 1);
   *d_y += t.pushforward * p;
}

template <typename T>
ValueAndPushforward<T, T> Sin_pushforward(T x, T d_x)
{
   return {::TMath::Sin(x), ::TMath::Cos(x) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> SinH_pushforward(T x, T d_x)
{
   return {::TMath::SinH(x), ::TMath::CosH(x) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Sq_pushforward(T x, T d_x)
{
   return {::TMath::Sq(x), 2 * x * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Sqrt_pushforward(T x, T d_x)
{
   return {::TMath::Sqrt(x), (0.5 / ::TMath::Sqrt(x)) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> Tan_pushforward(T x, T d_x)
{
   return {::TMath::Tan(x), (1. / ::TMath::Sq(::TMath::Cos(x))) * d_x};
}

template <typename T>
ValueAndPushforward<T, T> TanH_pushforward(T x, T d_x)
{
   return {::TMath::TanH(x), (1. / ::TMath::Sq(::TMath::CosH(x))) * d_x};
}

#ifdef WIN32
// Additional custom derivatives that can be removed
// after Issue #12108 in ROOT is resolved
// constexpr is removed
ValueAndPushforward<Double_t, Double_t> Pi_pushforward()
{
   return {3.1415926535897931, 0.};
}
// constexpr is removed
ValueAndPushforward<Double_t, Double_t> Ln10_pushforward()
{
   return {2.3025850929940459, 0.};
}
#endif
} // namespace TMath
} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_DERIVATOR
