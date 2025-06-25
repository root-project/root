// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2006

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Implementation of VectorUtil functions
//
// Created by: Lorenzo Moneta 22 Aug 2006
//
#include "Math/GenVector/VectorUtil.h"
#include "TMath.h"

namespace ROOT {

namespace Math {


double VectorUtil::Phi_0_2pi(double angle) {
   // returns phi angle in the interval (0,2*PI]
   if (angle <= 2. * TMath::Pi() && angle > 0)
      return angle;

   if ( angle > 0 ) {
      int n = static_cast<int>(angle / (2. * TMath::Pi()));
      angle -= 2. * TMath::Pi() * n;
   } else {
      int n = static_cast<int>(-(angle) / (2. * TMath::Pi()));
      angle += 2. * TMath::Pi() * (n + 1);
   }
   return angle;
}

double VectorUtil::Phi_mpi_pi(double angle) {
   // returns phi angle in the interval (-PI,PI]

   if (angle <= TMath::Pi() && angle > -TMath::Pi())
      return angle;

   if ( angle > 0 ) {
      int n = static_cast<int>((angle + TMath::Pi()) / (2. * TMath::Pi()));
      angle -= 2 * TMath::Pi() * n;
   } else {
      int n = static_cast<int>(-(angle - TMath::Pi()) / (2. * TMath::Pi()));
      angle += 2 * TMath::Pi() * n;
   }
   return angle;
}



} //namespace Math
} //namespace ROOT
