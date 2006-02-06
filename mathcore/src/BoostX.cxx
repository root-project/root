       // @(#)root/mathcore:$Name:  $:$Id: BoostX.cpp,v 1.4 2006/02/04 16:10:26 moneta Exp $
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

void
BoostX::SetComponents (Scalar bx ) {
  Scalar bp2 = bx*bx;
  if (bp2 >= 1) {
    GenVector_exception e ( 
      "Beta Vector supplied to set BoostX represents speed >= c");
    Throw(e);
    return;
  }    
  fBeta = bx;
  fGamma = 1.0 / std::sqrt(1.0 - bp2);
}

void
BoostX::GetComponents (Scalar& bx) const {
  bx = fBeta;
}

DisplacementVector3D< Cartesian3D<BoostX::Scalar> >
BoostX::BetaVector() const {
  return DisplacementVector3D< Cartesian3D<Scalar> > ( fBeta, 0.0, 0.0 );
}

void 
BoostX::GetLorentzRotation (Scalar r[]) const {
  r[LXX] = fGamma;        r[LXY] = 0.0;  r[LXZ] = 0.0;  r[LXT] = fGamma*fBeta;  
  r[LYX] = 0.0;           r[LYY] = 1.0;  r[LYZ] = 0.0;  r[LYT] = 0.0;  
  r[LZX] = 0.0;           r[LZY] = 0.0;  r[LZZ] = 1.0;  r[LZT] = 0.0;  
  r[LTX] = fGamma*fBeta;  r[LTY] = 0.0;  r[LTZ] = 0.0;  r[LTT] = fGamma;  
}

void 
BoostX::
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
BoostX::
operator() (const LorentzVector< PxPyPzE4D<double> > & v) const {
  Scalar x = v.Px();
  Scalar t = v.E();
  return LorentzVector< PxPyPzE4D<double> > 
    ( fGamma*x       + fGamma*fBeta*t 
      ,  v.Py()
      ,  v.Pz()
    , fGamma*fBeta*x + fGamma*t );
}

void 
BoostX::
Invert() { fBeta = -fBeta; }

BoostX
BoostX::
Inverse() const {
  BoostX I(*this);
  I.Invert();
  return I; 
}

// ========== I/O =====================

std::ostream & operator<< (std::ostream & os, const BoostX & b) {
  os << " BoostX( beta: " << b.Beta() << ", gamma: " << b.Gamma() << " ) ";
  return os;
}

} //namespace Math
} //namespace ROOT
