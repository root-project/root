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

namespace custom_derivatives {
  template <typename T>
  T Abs_darg0(T d) {
    return (d < 0) ? -1 : 1;
  }

  template <typename T>
  Double_t ACos_darg0(T d) {
    return -1./TMath::Sqrt(1 - d * d);
  }

  template <typename T>
  Double_t ACosH_darg0(T d) {
    return 1. / TMath::Sqrt(d * d - 1);
  }

  template <typename T>
  Double_t ASin_darg0(T d) {
    return 1. / TMath::Sqrt(1 - d * d);
  }

  template <typename T>
  Double_t ASinH_darg0(T d) {
    return 1. / TMath::Sqrt(d * d + 1);
  }

  template <typename T>
  Double_t ATan_darg0(T d) {
    return 1. / (d * d + 1);
  }

  template <typename T>
  Double_t ATanH_darg0(T d) {
    return 1. / (1 - d * d);
  }

  template <typename T>
  T Cos_darg0(T d) {
    return -TMath::Sin(d);
  }

  template <typename T>
  T CosH_darg0(T d) {
    return TMath::SinH(d);
  }

  template <typename T>
  Double_t Erf_darg0(T d) {
    return 2 * TMath::Exp(-d * d) / TMath::Sqrt(TMath::Pi());
  }

  template <typename T>
  Double_t Erfc_darg0(T d) { 
    return -Erf_darg(d);
  }

  template <typename T>
  Double_t Exp_darg0(T d) {
    return TMath::Exp(d);
  }

  template <typename T>
  T Hypot_darg0(T x, T y) {
    return x / TMath::Hypot(x, y);
  }

  template <typename T>
  T Hypot_darg1(T x, T y) {
    return y / TMath::Hypot(x, y);
  }
    
  template <typename T>
  void Hypot_grad(T x, T y, T* result) {
    T h = TMath::Hypot(x, y);
    result[0] += x / h;
    result[1] += y / h;
  }

  template <typename T>
  Double_t Log_darg0(T d) {
    return 1. / d;
  }

  template <typename T>
  Double_t Log10_darg0(T d) {
    return Log_darg0(d) / TMath::Ln10();
  } 

  template <typename T>
  Double_t Log2_darg0(T d) {
    return Log_darg0(d) / TMath::Log(2);
  } 

  template <typename T>
  T Max_darg0(T a, T b) {
    return (a >= b) ? 1 : 0;
  }

  template <typename T>
  T Max_darg1(T a, T b) {
    return (a >= b) ? 0 : 1;
  }

  template <typename T>
  void Max_grad(T a, T b, T* result) {
    if (a >= b)
      result[0] += 1;
    else
      result[1] += 1;
  }

  template <typename T>
  T Min_darg0(T a, T b) {
    return (a <= b) ? 1 : 0;
  }

  template <typename T>
  T Min_darg1(T a, T b) {
    return (a <= b) ? 0 : 1;
  }

  template <typename T>
  void Min_grad(T a, T b, T* result) {
    if (a <= b)
      result[0] += 1;
    else
      result[1] += 1;
  }

  template <typename T>
  T Power_darg0(T x, T y) {
    return y * TMath::Power(x, y - 1);
  }

  template <typename T>
  Double_t Power_darg1(T x, T y) {
    return TMath::Power(x, y) * TMath::Log(x);
  }

  template <typename T>
  Double_t Power_grad(T x, T y, Double_t* result) {
    T t = TMath::Power(x, y - 1);
    result[0] += y * t;
    result[1] += x * t * TMath::Log(x);
  }

  template <typename T>
  Double_t Sin_darg0(T d) {
    return TMath::Cos(d);
  }

  template <typename T>
  Double_t SinH_darg0(T d) {
    return TMath::CosH(d);
  }

  template <typename T>
  T Sq_darg0(T d) {
    return 2 * d;
  }

  template <typename T>
  Double_t Sqrt_darg0(T d) {
    return 0.5 / TMath::Sqrt(d);
  }

  template <typename T>
  Double_t Tan_darg0(T d) {
    return 1./ TMath::Sq(TMath::Cos(d));
  }

  template <typename T>
  Double_t TanH_darg0(T d) {
    return 1./ TMath::Sq(TMath::CosH(d));
  }
}
#endif // CLAD_DERIVATOR
