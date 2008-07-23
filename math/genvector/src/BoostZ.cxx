// @(#)root/mathcore:$Id$
// Authors:  M. Fischler  2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class BoostZ, a 4x4 symmetric matrix representation of
// an axial Lorentz transformation
//
// Created by: Mark Fischler Mon Nov 1  2005
//
#include "Math/GenVector/BoostZ.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/GenVector_exception.h"

#include <cmath>
#include <algorithm>

namespace ROOT {

namespace Math {

BoostZ::BoostZ() : fBeta(0.0), fGamma(1.0) {}

void BoostZ::SetComponents (Scalar bz) {
   // set component
   Scalar bp2 = bz*bz;
   if (bp2 >= 1) {
      GenVector::Throw ( 
                              "Beta Vector supplied to set BoostZ represents speed >= c");
      return;
   }    
   fBeta = bz;
   fGamma = 1.0 / std::sqrt(1.0 - bp2);
}

void BoostZ::GetComponents (Scalar& bz) const {
   // get component
   bz = fBeta;
}

DisplacementVector3D< Cartesian3D<BoostZ::Scalar> >
BoostZ::BetaVector() const {
   // return beta vector
   return DisplacementVector3D< Cartesian3D<Scalar> >
   ( 0.0, 0.0, fBeta );
}

void BoostZ::GetLorentzRotation (Scalar r[]) const {
   // get corresponding LorentzRotation
   r[kLXX] = 1.0;  r[kLXY] = 0.0;  r[kLXZ] = 0.0;           r[kLXT] = 0.0   ;  
   r[kLYX] = 0.0;  r[kLYY] = 1.0;  r[kLYZ] = 0.0;           r[kLYT] = 0.0   ;  
   r[kLZX] = 0.0;  r[kLZY] = 0.0;  r[kLZZ] = fGamma;        r[kLZT] = fGamma*fBeta;  
   r[kLTX] = 0.0;  r[kLTY] = 0.0;  r[kLTZ] = fGamma*fBeta;  r[kLTT] = fGamma;
}

void BoostZ::Rectify() {
   // Assuming the representation of this is close to a true Lorentz Rotation,
   // but may have drifted due to round-off error from many operations,
   // this forms an "exact" orthosymplectic matrix for the Lorentz Rotation
   // again.
   
   if (fGamma <= 0) {	
      GenVector::Throw ( 
                              "Attempt to rectify a boost with non-positive gamma");
      return;
   }    
   Scalar beta = fBeta;
   if ( beta >= 1 ) {			    
      beta /= ( beta * ( 1.0 + 1.0e-16 ) );  
   }
   SetComponents ( beta );
}

LorentzVector< PxPyPzE4D<double> >
BoostZ::operator() (const LorentzVector< PxPyPzE4D<double> > & v) const {
   // apply boost to a LV
   Scalar z = v.Pz();
   Scalar t = v.E();
   return LorentzVector< PxPyPzE4D<double> > 
      (  v.Px()
         ,  v.Py()
         , fGamma*z        + fGamma*fBeta*t
         , fGamma*fBeta*z  + fGamma*t );
}

void BoostZ::Invert() {
   // invert
   fBeta = -fBeta;
}

BoostZ BoostZ::Inverse() const {
   // return an inverse boostZ
   BoostZ tmp(*this);
   tmp.Invert();
   return tmp; 
}

// ========== I/O =====================

std::ostream & operator<< (std::ostream & os, const BoostZ & b) {
   os << " BoostZ( beta: " << b.Beta() << ", gamma: " << b.Gamma() << " ) ";
   return os;
}

} //namespace Math
} //namespace ROOT
