       // @(#)root/mathcore:$Id$
// Authors:  M. Fischler  2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class BoostX, a 4x4 symmetric matrix representation of
// an axial Lorentz transformation
//
// Created by: Mark Fischler Mon Nov 1  2005
//
#include "Math/GenVector/BoostX.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/GenVector_exception.h"

#include <cmath>
#include <algorithm>

namespace ROOT {

namespace Math {


BoostX::BoostX() : fBeta(0.0), fGamma(1.0) {}

void BoostX::SetComponents (Scalar bx ) {
   // set component
   Scalar bp2 = bx*bx;
   if (bp2 >= 1) {
      GenVector::Throw ( 
                              "Beta Vector supplied to set BoostX represents speed >= c");
      return;
   }    
   fBeta = bx;
   fGamma = 1.0 / std::sqrt(1.0 - bp2);
}

void BoostX::GetComponents (Scalar& bx) const {
   // get component
   bx = fBeta;
}

DisplacementVector3D< Cartesian3D<BoostX::Scalar> >
BoostX::BetaVector() const {
   // return beta vector
   return DisplacementVector3D< Cartesian3D<Scalar> > ( fBeta, 0.0, 0.0 );
}

void BoostX::GetLorentzRotation (Scalar r[]) const {
   // get corresponding LorentzRotation
   r[kLXX] = fGamma;        r[kLXY] = 0.0;  r[kLXZ] = 0.0;  r[kLXT] = fGamma*fBeta;  
   r[kLYX] = 0.0;           r[kLYY] = 1.0;  r[kLYZ] = 0.0;  r[kLYT] = 0.0;  
   r[kLZX] = 0.0;           r[kLZY] = 0.0;  r[kLZZ] = 1.0;  r[kLZT] = 0.0;  
   r[kLTX] = fGamma*fBeta;  r[kLTY] = 0.0;  r[kLTZ] = 0.0;  r[kLTT] = fGamma;  
}

void BoostX::Rectify() {
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
BoostX::operator() (const LorentzVector< PxPyPzE4D<double> > & v) const {
   // apply boost to a LV
   Scalar x = v.Px();
   Scalar t = v.E();
   return LorentzVector< PxPyPzE4D<double> > 
      ( fGamma*x       + fGamma*fBeta*t 
        ,  v.Py()
        ,  v.Pz()
        , fGamma*fBeta*x + fGamma*t );
}

void BoostX::Invert() { 
   // invert
   fBeta = -fBeta; 
}

BoostX BoostX::Inverse() const {
   // return an inverse boostX
   BoostX tmp(*this);
   tmp.Invert();
   return tmp; 
}

// ========== I/O =====================

std::ostream & operator<< (std::ostream & os, const BoostX & b) {
   os << " BoostX( beta: " << b.Beta() << ", gamma: " << b.Gamma() << " ) ";
   return os;
}

} //namespace Math
} //namespace ROOT
