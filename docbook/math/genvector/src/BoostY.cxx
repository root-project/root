// @(#)root/mathcore:$Id$
// Authors:  M. Fischler  2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class BoostY, a 4x4 symmetric matrix representation of
// an axial Lorentz transformation
//
// Created by: Mark Fischler Mon Nov 1  2005
//
#include "Math/GenVector/BoostY.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/GenVector_exception.h"

#include <cmath>
#include <algorithm>

namespace ROOT {

namespace Math {

BoostY::BoostY() : fBeta(0.0), fGamma(1.0) {}

void BoostY::SetComponents (Scalar by) {
   // set component
   Scalar bp2 = by*by;
   if (bp2 >= 1) {
      GenVector::Throw( 
                              "Beta Vector supplied to set BoostY represents speed >= c");
      return;
   }    
   fBeta = by;
   fGamma = 1.0 / std::sqrt(1.0-bp2);
}

void BoostY::GetComponents (Scalar& by) const {
   // get component
   by = fBeta;
}

DisplacementVector3D< Cartesian3D<BoostY::Scalar> >
BoostY::BetaVector() const {
   // return beta vector
   return DisplacementVector3D< Cartesian3D<Scalar> > ( 0.0, fBeta, 0.0 );
}

void BoostY::GetLorentzRotation (Scalar r[]) const {
   // get corresponding LorentzRotation
   r[kLXX] = 1.0;  r[kLXY] = 0.0;           r[kLXZ] = 0.0;  r[kLXT] = 0.0;  
   r[kLYX] = 0.0;  r[kLYY] = fGamma;        r[kLYZ] = 0.0;  r[kLYT] = fGamma*fBeta; 
   r[kLZX] = 0.0;  r[kLZY] = 0.0;           r[kLZZ] = 1.0;  r[kLZT] = 0.0;  
   r[kLTX] = 0.0;  r[kLTY] = fGamma*fBeta;  r[kLTZ] = 0.0;  r[kLTT] = fGamma;  
}

void BoostY::Rectify() {
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
BoostY::operator() (const LorentzVector< PxPyPzE4D<double> > & v) const {
   // apply boost to a LV
   Scalar y = v.Py();
   Scalar t = v.E();
   return LorentzVector< PxPyPzE4D<double> > 
      (  v.Px()
         , fGamma*y       + fGamma*fBeta*t
         ,  v.Pz()
         , fGamma*fBeta*y + fGamma*t );
}

void BoostY::Invert() {
   // invert Boost
   fBeta = -fBeta;
}

BoostY BoostY::Inverse() const {
   // return inverse
   BoostY tmp(*this);
   tmp.Invert();
   return tmp; 
}

// ========== I/O =====================

std::ostream & operator<< (std::ostream & os, const BoostY & b) {
   os << " BoostY( beta: " << b.Beta() << ", gamma: " << b.Gamma() << " ) ";
   return os;
}

} //namespace Math
} //namespace ROOT
