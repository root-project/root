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


namespace ROOT {
namespace Math {

inline void landau_pdf_pullback(double x, double xi, double x0, double d_out, double *d_x, double *d_xi, double *d_x0)
{
   if (xi <= 0) {
      return;
   }
   // clang-format off
   static double p1[5] = {0.4259894875,-0.1249762550, 0.03984243700, -0.006298287635,   0.001511162253};
   static double q1[5] = {1.0         ,-0.3388260629, 0.09594393323, -0.01608042283,    0.003778942063};

   static double p2[5] = {0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411,   0.0001283617211};
   static double q2[5] = {1.0         , 0.7428795082, 0.3153932961,   0.06694219548,    0.008790609714};

   static double p3[5] = {0.1788544503, 0.09359161662,0.006325387654, 0.00006611667319,-0.000002031049101};
   static double q3[5] = {1.0         , 0.6097809921, 0.2560616665,   0.04746722384,    0.006957301675};

   static double p4[5] = {0.9874054407, 118.6723273,  849.2794360,   -743.7792444,      427.0262186};
   static double q4[5] = {1.0         , 106.8615961,  337.6496214,    2016.712389,      1597.063511};

   static double p5[5] = {1.003675074,  167.5702434,  4789.711289,    21217.86767,     -22324.94910};
   static double q5[5] = {1.0         , 156.9424537,  3745.310488,    9834.698876,      66924.28357};

   static double p6[5] = {1.000827619,  664.9143136,  62972.92665,    475554.6998,     -5743609.109};
   static double q6[5] = {1.0         , 651.4101098,  56974.73333,    165917.4725,     -2815759.939};

   static double a1[3] = {0.04166666667,-0.01996527778, 0.02709538966};

   static double a2[2] = {-1.845568670,-4.284640743};
   // clang-format on
   const double _const0 = 0.3989422803;
   double v = (x - x0) / xi;
   double _d_v = 0;
   double _d_denlan = 0;
   if (v < -5.5) {
      double _t0;
      double u = ::std::exp(v + 1.);
      double _d_u = 0;
      if (u >= 1.e-10) {
         const double ue = ::std::exp(-1 / u);
         const double us = ::std::sqrt(u);
         double _t3;
         double _d_ue = 0;
         double _d_us = 0;
         double denlan = _const0 * (ue / us) * (1 + (a1[0] + (a1[1] + a1[2] * u) * u) * u);
         _d_denlan += d_out / xi;
         *d_xi += d_out * -(denlan / (xi * xi));
         denlan = _t3;
         double _r_d3 = _d_denlan;
         _d_denlan -= _r_d3;
         _d_ue += _const0 * _r_d3 * (1 + (a1[0] + (a1[1] + a1[2] * u) * u) * u) / us;
         double _r5 = _const0 * _r_d3 * (1 + (a1[0] + (a1[1] + a1[2] * u) * u) * u) * -(ue / (us * us));
         _d_us += _r5;
         _d_u += a1[2] * _const0 * (ue / us) * _r_d3 * u * u;
         _d_u += (a1[1] + a1[2] * u) * _const0 * (ue / us) * _r_d3 * u;
         _d_u += (a1[0] + (a1[1] + a1[2] * u) * u) * _const0 * (ue / us) * _r_d3;
         double _r_d2 = _d_us;
         _d_us -= _r_d2;
         double _r4 = 0;
         _r4 += _r_d2 * clad::custom_derivatives::sqrt_pushforward(u, 1.).pushforward;
         _d_u += _r4;
         double _r_d1 = _d_ue;
         _d_ue -= _r_d1;
         double _r2 = 0;
         _r2 += _r_d1 * clad::custom_derivatives::exp_pushforward(-1 / u, 1.).pushforward;
         double _r3 = _r2 * -(-1 / (u * u));
         _d_u += _r3;
      }
      u = _t0;
      double _r_d0 = _d_u;
      _d_u -= _r_d0;
      double _r1 = 0;
      _r1 += _r_d0 * clad::custom_derivatives::exp_pushforward(v + 1., 1.).pushforward;
      _d_v += _r1;
   } else if (v < -1) {
      double _t4;
      double u = ::std::exp(-v - 1);
      double _d_u = 0;
      double _t5;
      double _t8 = ::std::exp(-u);
      double _t7 = ::std::sqrt(u);
      double _t6 = (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * v);
      double denlan = _t8 * _t7 * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) / _t6;
      _d_denlan += d_out / xi;
      *d_xi += d_out * -(denlan / (xi * xi));
      denlan = _t5;
      double _r_d5 = _d_denlan;
      _d_denlan -= _r_d5;
      double _r7 = 0;
      _r7 += _r_d5 / _t6 * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) * _t7 *
             clad::custom_derivatives::exp_pushforward(-u, 1.).pushforward;
      _d_u += -_r7;
      double _r8 = 0;
      _r8 += _t8 * _r_d5 / _t6 * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) *
             clad::custom_derivatives::sqrt_pushforward(u, 1.).pushforward;
      _d_u += _r8;
      _d_v += p1[4] * _t8 * _t7 * _r_d5 / _t6 * v * v * v;
      _d_v += (p1[3] + p1[4] * v) * _t8 * _t7 * _r_d5 / _t6 * v * v;
      _d_v += (p1[2] + (p1[3] + p1[4] * v) * v) * _t8 * _t7 * _r_d5 / _t6 * v;
      _d_v += (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * _t8 * _t7 * _r_d5 / _t6;
      double _r9 = _r_d5 * -(_t8 * _t7 * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) / (_t6 * _t6));
      _d_v += q1[4] * _r9 * v * v * v;
      _d_v += (q1[3] + q1[4] * v) * _r9 * v * v;
      _d_v += (q1[2] + (q1[3] + q1[4] * v) * v) * _r9 * v;
      _d_v += (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * _r9;
      u = _t4;
      double _r_d4 = _d_u;
      _d_u -= _r_d4;
      double _r6 = 0;
      _r6 += _r_d4 * clad::custom_derivatives::exp_pushforward(-v - 1, 1.).pushforward;
      _d_v += -_r6;
   } else if (v < 1) {
      double _t9;
      double _t10 = (q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4] * v) * v) * v) * v);
      double denlan = (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4] * v) * v) * v) * v) / _t10;
      _d_denlan += d_out / xi;
      *d_xi += d_out * -(denlan / (xi * xi));
      denlan = _t9;
      double _r_d6 = _d_denlan;
      _d_denlan -= _r_d6;
      _d_v += p2[4] * _r_d6 / _t10 * v * v * v;
      _d_v += (p2[3] + p2[4] * v) * _r_d6 / _t10 * v * v;
      _d_v += (p2[2] + (p2[3] + p2[4] * v) * v) * _r_d6 / _t10 * v;
      _d_v += (p2[1] + (p2[2] + (p2[3] + p2[4] * v) * v) * v) * _r_d6 / _t10;
      double _r10 = _r_d6 * -((p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4] * v) * v) * v) * v) / (_t10 * _t10));
      _d_v += q2[4] * _r10 * v * v * v;
      _d_v += (q2[3] + q2[4] * v) * _r10 * v * v;
      _d_v += (q2[2] + (q2[3] + q2[4] * v) * v) * _r10 * v;
      _d_v += (q2[1] + (q2[2] + (q2[3] + q2[4] * v) * v) * v) * _r10;
   } else if (v < 5) {
      double _t11;
      double _t12 = (q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4] * v) * v) * v) * v);
      double denlan = (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4] * v) * v) * v) * v) / _t12;
      _d_denlan += d_out / xi;
      *d_xi += d_out * -(denlan / (xi * xi));
      denlan = _t11;
      double _r_d7 = _d_denlan;
      _d_denlan -= _r_d7;
      _d_v += p3[4] * _r_d7 / _t12 * v * v * v;
      _d_v += (p3[3] + p3[4] * v) * _r_d7 / _t12 * v * v;
      _d_v += (p3[2] + (p3[3] + p3[4] * v) * v) * _r_d7 / _t12 * v;
      _d_v += (p3[1] + (p3[2] + (p3[3] + p3[4] * v) * v) * v) * _r_d7 / _t12;
      double _r11 = _r_d7 * -((p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4] * v) * v) * v) * v) / (_t12 * _t12));
      _d_v += q3[4] * _r11 * v * v * v;
      _d_v += (q3[3] + q3[4] * v) * _r11 * v * v;
      _d_v += (q3[2] + (q3[3] + q3[4] * v) * v) * _r11 * v;
      _d_v += (q3[1] + (q3[2] + (q3[3] + q3[4] * v) * v) * v) * _r11;
   } else if (v < 12) {
      double u = 1 / v;
      double _d_u = 0;
      double _t14;
      double _t15 = (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u);
      double denlan = u * u * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) / _t15;
      _d_denlan += d_out / xi;
      *d_xi += d_out * -(denlan / (xi * xi));
      denlan = _t14;
      double _r_d9 = _d_denlan;
      _d_denlan -= _r_d9;
      _d_u += _r_d9 / _t15 * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) * u;
      _d_u += u * _r_d9 / _t15 * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u);
      _d_u += p4[4] * u * u * _r_d9 / _t15 * u * u * u;
      _d_u += (p4[3] + p4[4] * u) * u * u * _r_d9 / _t15 * u * u;
      _d_u += (p4[2] + (p4[3] + p4[4] * u) * u) * u * u * _r_d9 / _t15 * u;
      _d_u += (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u * u * _r_d9 / _t15;
      double _r13 = _r_d9 * -(u * u * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) / (_t15 * _t15));
      _d_u += q4[4] * _r13 * u * u * u;
      _d_u += (q4[3] + q4[4] * u) * _r13 * u * u;
      _d_u += (q4[2] + (q4[3] + q4[4] * u) * u) * _r13 * u;
      _d_u += (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * _r13;
      double _r_d8 = _d_u;
      _d_u -= _r_d8;
      double _r12 = _r_d8 * -(1 / (v * v));
      _d_v += _r12;
   } else if (v < 50) {
      double u = 1 / v;
      double _d_u = 0;
      double _t17;
      double _t18 = (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u);
      double denlan = u * u * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) / _t18;
      _d_denlan += d_out / xi;
      *d_xi += d_out * -(denlan / (xi * xi));
      denlan = _t17;
      double _r_d11 = _d_denlan;
      _d_denlan -= _r_d11;
      _d_u += _r_d11 / _t18 * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) * u;
      _d_u += u * _r_d11 / _t18 * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u);
      _d_u += p5[4] * u * u * _r_d11 / _t18 * u * u * u;
      _d_u += (p5[3] + p5[4] * u) * u * u * _r_d11 / _t18 * u * u;
      _d_u += (p5[2] + (p5[3] + p5[4] * u) * u) * u * u * _r_d11 / _t18 * u;
      _d_u += (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u * u * _r_d11 / _t18;
      double _r15 = _r_d11 * -(u * u * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) / (_t18 * _t18));
      _d_u += q5[4] * _r15 * u * u * u;
      _d_u += (q5[3] + q5[4] * u) * _r15 * u * u;
      _d_u += (q5[2] + (q5[3] + q5[4] * u) * u) * _r15 * u;
      _d_u += (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * _r15;
      double _r_d10 = _d_u;
      _d_u -= _r_d10;
      double _r14 = _r_d10 * -(1 / (v * v));
      _d_v += _r14;
   } else if (v < 300) {
      double _t19;
      double u = 1 / v;
      double _d_u = 0;
      double _t20;
      double _t21 = (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u);
      double denlan = u * u * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) / _t21;
      _d_denlan += d_out / xi;
      *d_xi += d_out * -(denlan / (xi * xi));
      denlan = _t20;
      double _r_d13 = _d_denlan;
      _d_denlan -= _r_d13;
      _d_u += _r_d13 / _t21 * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) * u;
      _d_u += u * _r_d13 / _t21 * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u);
      _d_u += p6[4] * u * u * _r_d13 / _t21 * u * u * u;
      _d_u += (p6[3] + p6[4] * u) * u * u * _r_d13 / _t21 * u * u;
      _d_u += (p6[2] + (p6[3] + p6[4] * u) * u) * u * u * _r_d13 / _t21 * u;
      _d_u += (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u * u * _r_d13 / _t21;
      double _r17 = _r_d13 * -(u * u * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) / (_t21 * _t21));
      _d_u += q6[4] * _r17 * u * u * u;
      _d_u += (q6[3] + q6[4] * u) * _r17 * u * u;
      _d_u += (q6[2] + (q6[3] + q6[4] * u) * u) * _r17 * u;
      _d_u += (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * _r17;
      u = _t19;
      double _r_d12 = _d_u;
      _d_u -= _r_d12;
      double _r16 = _r_d12 * -(1 / (v * v));
      _d_v += _r16;
   } else {
      double _t22;
      double _t25 = ::std::log(v);
      double _t24 = (v + 1);
      double _t23 = (v - v * _t25 / _t24);
      double u = 1 / _t23;
      double _d_u = 0;
      double _t26;
      double denlan = u * u * (1 + (a2[0] + a2[1] * u) * u);
      _d_denlan += d_out / xi;
      *d_xi += d_out * -(denlan / (xi * xi));
      denlan = _t26;
      double _r_d15 = _d_denlan;
      _d_denlan -= _r_d15;
      _d_u += _r_d15 * (1 + (a2[0] + a2[1] * u) * u) * u;
      _d_u += u * _r_d15 * (1 + (a2[0] + a2[1] * u) * u);
      _d_u += a2[1] * u * u * _r_d15 * u;
      _d_u += (a2[0] + a2[1] * u) * u * u * _r_d15;
      u = _t22;
      double _r_d14 = _d_u;
      _d_u -= _r_d14;
      double _r18 = _r_d14 * -(1 / (_t23 * _t23));
      _d_v += _r18;
      _d_v += -_r18 / _t24 * _t25;
      double _r19 = 0;
      _r19 += v * -_r18 / _t24 * clad::custom_derivatives::log_pushforward(v, 1.).pushforward;
      _d_v += _r19;
      double _r20 = -_r18 * -(v * _t25 / (_t24 * _t24));
      _d_v += _r20;
   }
   *d_x += _d_v / xi;
   *d_x0 += -_d_v / xi;
   double _r0 = _d_v * -((x - x0) / (xi * xi));
   *d_xi += _r0;
}

inline void landau_cdf_pullback(double x, double xi, double x0, double d_out, double *d_x, double *d_xi, double *d_x0)
{
   // clang-format off
   static double p1[5] = {0.2514091491e+0,-0.6250580444e-1, 0.1458381230e-1,-0.2108817737e-2, 0.7411247290e-3};
   static double q1[5] = {1.0            ,-0.5571175625e-2, 0.6225310236e-1,-0.3137378427e-2, 0.1931496439e-2};

   static double p2[4] = {0.2868328584e+0, 0.3564363231e+0, 0.1523518695e+0, 0.2251304883e-1};
   static double q2[4] = {1.0            , 0.6191136137e+0, 0.1720721448e+0, 0.2278594771e-1};

   static double p3[4] = {0.2868329066e+0, 0.3003828436e+0, 0.9950951941e-1, 0.8733827185e-2};
   static double q3[4] = {1.0            , 0.4237190502e+0, 0.1095631512e+0, 0.8693851567e-2};

   static double p4[4] = {0.1000351630e+1, 0.4503592498e+1, 0.1085883880e+2, 0.7536052269e+1};
   static double q4[4] = {1.0            , 0.5539969678e+1, 0.1933581111e+2, 0.2721321508e+2};

   static double p5[4] = {0.1000006517e+1, 0.4909414111e+2, 0.8505544753e+2, 0.1532153455e+3};
   static double q5[4] = {1.0            , 0.5009928881e+2, 0.1399819104e+3, 0.4200002909e+3};

   static double p6[4] = {0.1000000983e+1, 0.1329868456e+3, 0.9162149244e+3,-0.9605054274e+3};
   static double q6[4] = {1.0            , 0.1339887843e+3, 0.1055990413e+4, 0.5532224619e+3};

   static double a1[4] = {0              ,-0.4583333333e+0, 0.6675347222e+0,-0.1641741416e+1};
   static double a2[4] = {0              , 1.0            ,-0.4227843351e+0,-0.2043403138e+1};
   // clang-format on

   const double v = (x - x0) / xi;
   double _d_v = 0;
   if (v < -5.5) {
      double _d_u = 0;
      const double _const0 = 0.3989422803;
      double u = ::std::exp(v + 1);
      double _t3 = ::std::exp(-1. / u);
      double _t2 = ::std::sqrt(u);
      double _r2 = 0;
      _r2 += _const0 * d_out * (1 + (a1[1] + (a1[2] + a1[3] * u) * u) * u) * _t2 *
             clad::custom_derivatives::exp_pushforward(-1. / u, 1.).pushforward;
      double _r3 = _r2 * -(-1. / (u * u));
      _d_u += _r3;
      double _r4 = 0;
      _r4 += _const0 * _t3 * d_out * (1 + (a1[1] + (a1[2] + a1[3] * u) * u) * u) *
             clad::custom_derivatives::sqrt_pushforward(u, 1.).pushforward;
      _d_u += _r4;
      _d_u += a1[3] * _const0 * _t3 * _t2 * d_out * u * u;
      _d_u += (a1[2] + a1[3] * u) * _const0 * _t3 * _t2 * d_out * u;
      _d_u += (a1[1] + (a1[2] + a1[3] * u) * u) * _const0 * _t3 * _t2 * d_out;
      _d_v += _d_u * clad::custom_derivatives::exp_pushforward(v + 1, 1.).pushforward;
   } else if (v < -1) {
      double _d_u = 0;
      double u = ::std::exp(-v - 1);
      double _t8 = ::std::exp(-u);
      double _t7 = ::std::sqrt(u);
      double _t6 = (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * v);
      double _r6 = 0;
      _r6 += d_out / _t6 * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) / _t7 *
             clad::custom_derivatives::exp_pushforward(-u, 1.).pushforward;
      _d_u += -_r6;
      double _r7 = d_out / _t6 * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) * -(_t8 / (_t7 * _t7));
      double _r8 = 0;
      _r8 += _r7 * clad::custom_derivatives::sqrt_pushforward(u, 1.).pushforward;
      _d_u += _r8;
      _d_v += p1[4] * (_t8 / _t7) * d_out / _t6 * v * v * v;
      _d_v += (p1[3] + p1[4] * v) * (_t8 / _t7) * d_out / _t6 * v * v;
      _d_v += (p1[2] + (p1[3] + p1[4] * v) * v) * (_t8 / _t7) * d_out / _t6 * v;
      _d_v += (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * (_t8 / _t7) * d_out / _t6;
      double _r9 = d_out * -((_t8 / _t7) * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) / (_t6 * _t6));
      _d_v += q1[4] * _r9 * v * v * v;
      _d_v += (q1[3] + q1[4] * v) * _r9 * v * v;
      _d_v += (q1[2] + (q1[3] + q1[4] * v) * v) * _r9 * v;
      _d_v += (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * _r9;
      _d_v += -_d_u * clad::custom_derivatives::exp_pushforward(-v - 1, 1.).pushforward;
   } else if (v < 1) {
      double _t10 = (q2[0] + (q2[1] + (q2[2] + q2[3] * v) * v) * v);
      _d_v += p2[3] * d_out / _t10 * v * v;
      _d_v += (p2[2] + p2[3] * v) * d_out / _t10 * v;
      _d_v += (p2[1] + (p2[2] + p2[3] * v) * v) * d_out / _t10;
      double _r10 = d_out * -((p2[0] + (p2[1] + (p2[2] + p2[3] * v) * v) * v) / (_t10 * _t10));
      _d_v += q2[3] * _r10 * v * v;
      _d_v += (q2[2] + q2[3] * v) * _r10 * v;
      _d_v += (q2[1] + (q2[2] + q2[3] * v) * v) * _r10;
   } else if (v < 4) {
      double _t12 = (q3[0] + (q3[1] + (q3[2] + q3[3] * v) * v) * v);
      _d_v += p3[3] * d_out / _t12 * v * v;
      _d_v += (p3[2] + p3[3] * v) * d_out / _t12 * v;
      _d_v += (p3[1] + (p3[2] + p3[3] * v) * v) * d_out / _t12;
      double _r11 = d_out * -((p3[0] + (p3[1] + (p3[2] + p3[3] * v) * v) * v) / (_t12 * _t12));
      _d_v += q3[3] * _r11 * v * v;
      _d_v += (q3[2] + q3[3] * v) * _r11 * v;
      _d_v += (q3[1] + (q3[2] + q3[3] * v) * v) * _r11;
   } else if (v < 12) {
      double _d_u = 0;
      double u = 1. / v;
      double _t15 = (q4[0] + (q4[1] + (q4[2] + q4[3] * u) * u) * u);
      _d_u += p4[3] * d_out / _t15 * u * u;
      _d_u += (p4[2] + p4[3] * u) * d_out / _t15 * u;
      _d_u += (p4[1] + (p4[2] + p4[3] * u) * u) * d_out / _t15;
      double _r13 = d_out * -((p4[0] + (p4[1] + (p4[2] + p4[3] * u) * u) * u) / (_t15 * _t15));
      _d_u += q4[3] * _r13 * u * u;
      _d_u += (q4[2] + q4[3] * u) * _r13 * u;
      _d_u += (q4[1] + (q4[2] + q4[3] * u) * u) * _r13;
      double _r12 = _d_u * -(1. / (v * v));
      _d_v += _r12;
   } else if (v < 50) {
      double _d_u = 0;
      double u = 1. / v;
      double _t18 = (q5[0] + (q5[1] + (q5[2] + q5[3] * u) * u) * u);
      _d_u += p5[3] * d_out / _t18 * u * u;
      _d_u += (p5[2] + p5[3] * u) * d_out / _t18 * u;
      _d_u += (p5[1] + (p5[2] + p5[3] * u) * u) * d_out / _t18;
      double _r15 = d_out * -((p5[0] + (p5[1] + (p5[2] + p5[3] * u) * u) * u) / (_t18 * _t18));
      _d_u += q5[3] * _r15 * u * u;
      _d_u += (q5[2] + q5[3] * u) * _r15 * u;
      _d_u += (q5[1] + (q5[2] + q5[3] * u) * u) * _r15;
      double _r14 = _d_u * -(1. / (v * v));
      _d_v += _r14;
   } else if (v < 300) {
      double _d_u = 0;
      double u = 1. / v;
      double _t21 = (q6[0] + (q6[1] + (q6[2] + q6[3] * u) * u) * u);
      _d_u += p6[3] * d_out / _t21 * u * u;
      _d_u += (p6[2] + p6[3] * u) * d_out / _t21 * u;
      _d_u += (p6[1] + (p6[2] + p6[3] * u) * u) * d_out / _t21;
      double _r17 = d_out * -((p6[0] + (p6[1] + (p6[2] + p6[3] * u) * u) * u) / (_t21 * _t21));
      _d_u += q6[3] * _r17 * u * u;
      _d_u += (q6[2] + q6[3] * u) * _r17 * u;
      _d_u += (q6[1] + (q6[2] + q6[3] * u) * u) * _r17;
      double _r16 = _d_u * -(1. / (v * v));
      _d_v += _r16;
   } else {
      double _d_u = 0;
      double _t25 = ::std::log(v);
      double _t24 = (v + 1);
      double _t23 = (v - v * _t25 / _t24);
      double u = 1. / _t23;
      double _t26;
      _d_u += a2[3] * -d_out * u * u;
      _d_u += (a2[2] + a2[3] * u) * -d_out * u;
      _d_u += (a2[1] + (a2[2] + a2[3] * u) * u) * -d_out;
      double _r18 = _d_u * -(1. / (_t23 * _t23));
      _d_v += _r18;
      _d_v += -_r18 / _t24 * _t25;
      double _r19 = 0;
      _r19 += v * -_r18 / _t24 * clad::custom_derivatives::log_pushforward(v, 1.).pushforward;
      _d_v += _r19;
      double _r20 = -_r18 * -(v * _t25 / (_t24 * _t24));
      _d_v += _r20;
   }

   *d_x += _d_v / xi;
   *d_x0 += -_d_v / xi;
   *d_xi += _d_v * -((x - x0) / (xi * xi));
}

} // namespace Math
} // namespace ROOT

} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_DERIVATOR
