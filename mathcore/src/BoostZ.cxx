// @(#)root/mathcore:$Name:  $:$Id: BoostZ.cpp,v 1.1 2005/11/16 19:30:47 marafino Exp $
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

void
BoostZ::SetComponents (Scalar bz) {
  Scalar bp2 = bz*bz;
  if (bp2 >= 1) {
    GenVector_exception e ( 
      "Beta Vector supplied to set BoostZ represents speed >= c");
    Throw(e);
    return;
  }    
  fBeta = bz;
  fGamma = 1.0 / std::sqrt(1.0 - bp2);
}

void
BoostZ::GetComponents (Scalar& bz) const {
  bz = fBeta;
}

DisplacementVector3D< Cartesian3D<BoostZ::Scalar> >
BoostZ::BetaVector() const {
  return DisplacementVector3D< Cartesian3D<Scalar> >
  			( 0.0, 0.0, fBeta );
}

void 
BoostZ::GetLorentzRotation (Scalar r[]) const {
  r[LXX] = 0.0;  r[LXY] = 0.0;  r[LXZ] = 0.0;           r[LXT] = 0.0   ;  
  r[LYX] = 0.0;  r[LYY] = 0.0;  r[LYZ] = 0.0;           r[LYT] = 0.0   ;  
  r[LZX] = 0.0;  r[LZY] = 0.0;  r[LZZ] = fGamma;        r[LZT] = fGamma*fBeta;  
  r[LTX] = 0.0;  r[LTY] = 0.0;  r[LTZ] = fGamma*fBeta;  r[LTT] = fGamma;
}

void 
BoostZ::
Rectify() {
  // Assuming the representation of this is close to a true Lorentz Rotation,
  // but may have drifted due to round-off error from many operations,
  // this forms an "exact" orthosymplectic matrix for the Lorentz Rotation
  // again.

  if (fGamma <= 0) {	
    GenVector_exception e ( 
      "Attempt to rectify a boost with non-positive gamma");
    Throw(e);
    return;
  }    
  Scalar beta = fBeta;
  if ( beta >= 1 ) {			    
    beta /= ( beta * ( 1.0 + 1.0e-16 ) );  
  }
  SetComponents ( beta );
}

LorentzVector< PxPyPzE4D<double> >
BoostZ::
operator() (const LorentzVector< PxPyPzE4D<double> > & v) const {
  Scalar z = v.Pz();
  Scalar t = v.E();
  return LorentzVector< PxPyPzE4D<double> > 
    (  0.0
    ,  0.0
    , fGamma*z        + fGamma*fBeta*t
    , fGamma*fBeta*z  + fGamma*t );
}

void 
BoostZ::
Invert() {
  fBeta = -fBeta;
}

BoostZ
BoostZ::
Inverse() const {
  BoostZ I(*this);
  I.Invert();
  return I; 
}

} //namespace Math
} //namespace ROOT
