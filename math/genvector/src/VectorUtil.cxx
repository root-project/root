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

namespace ROOT {

namespace Math {


double VectorUtil::Phi_0_2pi(double angle) {
   // returns phi angle in the interval (0,2*PI]
   if ( angle <= 2.*M_PI && angle > 0 ) return angle;

   if ( angle > 0 ) {
      int n = static_cast<int>( angle/(2.*M_PI) );
      angle -= 2.*M_PI*n;
   } else {
      int n = static_cast<int>( -(angle)/(2.*M_PI) );
      angle += 2.*M_PI*(n+1);  
   }
   return angle;
}

double VectorUtil::Phi_mpi_pi(double angle) {
   // returns phi angle in the interval (-PI,PI]
   
   if ( angle <= M_PI && angle > -M_PI ) return angle;
   
   if ( angle > 0 ) {
      int n = static_cast<int>( (angle+M_PI)/(2.*M_PI) );
      angle -= 2*M_PI*n;
   } else {
      int n = static_cast<int>( -(angle-M_PI)/(2.*M_PI) );
      angle += 2*M_PI*n;  
   }
   return angle;
} 



} //namespace Math
} //namespace ROOT
