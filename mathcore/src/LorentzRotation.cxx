// @(#)root/mathcore:$Name:  $:$Id: LorentzRotation.cxx,v 1.3 2005/10/18 09:13:34 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class LorentzRotation, a 4x4 matrix representation of
// a general Lorentz transformation
//
// Created by: Mark Fischler Mon Aug 8  2005
//

#include "Math/GenVector/GenVectorIO.h"

#include "Math/GenVector/LorentzRotation.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/GenVector_exception.h"

#include <cmath>
#include <algorithm>

#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"

namespace ROOT {

  namespace Math {

LorentzRotation::LorentzRotation() {
  fM[XX] = 1.0;  fM[XY] = 0.0; fM[XZ] = 0.0; fM[XT] = 0.0;
  fM[YX] = 0.0;  fM[YY] = 1.0; fM[YZ] = 0.0; fM[YT] = 0.0;
  fM[ZX] = 0.0;  fM[ZY] = 0.0; fM[ZZ] = 1.0; fM[ZT] = 0.0;
  fM[TX] = 0.0;  fM[TY] = 0.0; fM[TZ] = 0.0; fM[TT] = 1.0;
}

LorentzRotation::LorentzRotation(Rotation3D  const & r) {  
  r.GetComponents ( fM[XX], fM[XY], fM[XZ],
                    fM[YX], fM[YY], fM[YZ],
                    fM[ZX], fM[ZY], fM[ZZ] );
   					     fM[XT] = 0.0;
   					     fM[YT] = 0.0;
   					     fM[ZT] = 0.0;
  fM[TX] = 0.0;  fM[TY] = 0.0; fM[TZ] = 0.0; fM[TT] = 1.0;
}

LorentzRotation::LorentzRotation(AxisAngle  const & a) {  
  const Rotation3D r(a);
  r.GetComponents ( fM[XX], fM[XY], fM[XZ],
                    fM[YX], fM[YY], fM[YZ],
                    fM[ZX], fM[ZY], fM[ZZ] );
   					     fM[XT] = 0.0;
   					     fM[YT] = 0.0;
   					     fM[ZT] = 0.0;
  fM[TX] = 0.0;  fM[TY] = 0.0; fM[TZ] = 0.0; fM[TT] = 1.0;
}

LorentzRotation::LorentzRotation(EulerAngles  const & e) {  
  const Rotation3D r(e);
  r.GetComponents ( fM[XX], fM[XY], fM[XZ],
                    fM[YX], fM[YY], fM[YZ],
                    fM[ZX], fM[ZY], fM[ZZ] );
   					     fM[XT] = 0.0;
   					     fM[YT] = 0.0;
   					     fM[ZT] = 0.0;
  fM[TX] = 0.0;  fM[TY] = 0.0; fM[TZ] = 0.0; fM[TT] = 1.0;
}

LorentzRotation::LorentzRotation(Quaternion  const & q) {  
  const Rotation3D r(q);
  r.GetComponents ( fM[XX], fM[XY], fM[XZ],
                    fM[YX], fM[YY], fM[YZ],
                    fM[ZX], fM[ZY], fM[ZZ] );
   					     fM[XT] = 0.0;
   					     fM[YT] = 0.0;
   					     fM[ZT] = 0.0;
  fM[TX] = 0.0;  fM[TY] = 0.0; fM[TZ] = 0.0; fM[TT] = 1.0;
}

LorentzRotation::LorentzRotation(RotationX  const & r) {  
  Scalar s = r.SinAngle();
  Scalar c = r.CosAngle();
  fM[XX] = 1.0;  fM[XY] = 0.0; fM[XZ] = 0.0; fM[XT] = 0.0;
  fM[YX] = 0.0;  fM[YY] =  c ; fM[YZ] = -s ; fM[YT] = 0.0;
  fM[ZX] = 0.0;  fM[ZY] =  s ; fM[ZZ] =  c ; fM[ZT] = 0.0;
  fM[TX] = 0.0;  fM[TY] = 0.0; fM[TZ] = 0.0; fM[TT] = 1.0;
}

LorentzRotation::LorentzRotation(RotationY  const & r) {  
  Scalar s = r.SinAngle();
  Scalar c = r.CosAngle();
  fM[XX] =  c ;  fM[XY] = 0.0; fM[XZ] =  s ; fM[XT] = 0.0;
  fM[YX] = 0.0;  fM[YY] = 1.0; fM[YZ] = 0.0; fM[YT] = 0.0;
  fM[ZX] = -s ;  fM[ZY] = 0.0; fM[ZZ] =  c ; fM[ZT] = 0.0;
  fM[TX] = 0.0;  fM[TY] = 0.0; fM[TZ] = 0.0; fM[TT] = 1.0;
}

LorentzRotation::LorentzRotation(RotationZ  const & r) {  
  Scalar s = r.SinAngle();
  Scalar c = r.CosAngle();
  fM[XX] =  c ;  fM[XY] = -s ; fM[XZ] = 0.0; fM[XT] = 0.0;
  fM[YX] =  s ;  fM[YY] =  c ; fM[YZ] = 0.0; fM[YT] = 0.0;
  fM[ZX] = 0.0;  fM[ZY] = 0.0; fM[ZZ] = 1.0; fM[ZT] = 0.0;
  fM[TX] = 0.0;  fM[TY] = 0.0; fM[TZ] = 0.0; fM[TT] = 1.0;
}

void 
LorentzRotation::
Rectify() {
  // Assuming the representation of this is close to a true Lorentz Rotation,
  // but may have drifted due to round-off error from many operations,
  // this forms an "exact" orthosymplectic matrix for the Lorentz Rotation
  // again.
 
  typedef LorentzVector< PxPyPzE4D<Scalar> > FourVector;
  if (fM[TT] <= 0) {
    GenVector_exception e ( 
      "LorentzRotation:Rectify(): Non-positive TT component - cannot rectify");
    Throw(e);
    return;
  }  
  FourVector t ( fM[TX], fM[TY], fM[TZ], fM[TT] );
  Scalar m2 = t.M2();
  if ( m2 <= 0 ) {
    GenVector_exception e ( 
      "LorentzRotation:Rectify(): Non-timelike time row - cannot rectify");
    Throw(e);
    return;
  }
  t /= std::sqrt(m2);
  FourVector z ( fM[ZX], fM[ZY], fM[ZZ], fM[ZT] );
  z = z - z.Dot(t)*t;
  m2 = z.M2();
  if ( m2 >= 0 ) {
    GenVector_exception e ( 
      "LorentzRotation:Rectify(): Non-spacelike Z row projection - "
      "cannot rectify");
    Throw(e);
    return;
  }
  z /= std::sqrt(-m2);
  FourVector y ( fM[YX], fM[YY], fM[YZ], fM[YT] );
  y = y - y.Dot(t)*t - y.Dot(z)*z;
  m2 = y.M2();
  if ( m2 >= 0 ) {
    GenVector_exception e ( 
      "LorentzRotation:Rectify(): Non-spacelike Y row projection - "
      "cannot rectify");
    Throw(e);
    return;
  }
  y /= std::sqrt(-m2);
  FourVector x ( fM[XX], fM[XY], fM[XZ], fM[XT] );
  x = x - x.Dot(t)*t - x.Dot(z)*z - x.Dot(y)*y;
  m2 = x.M2();
  if ( m2 >= 0 ) {
    GenVector_exception e ( 
      "LorentzRotation:Rectify(): Non-spacelike X row projection - "
      "cannot rectify");
    Throw(e);
    return;
  }
  x /= std::sqrt(-m2);
}

LorentzVector< PxPyPzE4D<double> >
LorentzRotation::
operator() (const LorentzVector< PxPyPzE4D<double> > & v) const {
  Scalar x = v.Px();
  Scalar y = v.Py();
  Scalar z = v.Pz();
  Scalar t = v.E();
  return LorentzVector< PxPyPzE4D<double> > 
    ( fM[XX]*x + fM[XY]*y + fM[XZ]*z + fM[XT]*t 
    , fM[YX]*x + fM[YY]*y + fM[YZ]*z + fM[YT]*t
    , fM[ZX]*x + fM[ZY]*y + fM[ZZ]*z + fM[ZT]*t
    , fM[TX]*x + fM[TY]*y + fM[TZ]*z + fM[TT]*t );
}

void 
LorentzRotation::
Invert() {
  Scalar temp;
  temp = fM[XY]; fM[XY] =  fM[YX]; fM[YX] =  temp;  
  temp = fM[XZ]; fM[XZ] =  fM[ZX]; fM[ZX] =  temp;  
  temp = fM[YZ]; fM[YZ] =  fM[ZY]; fM[ZY] =  temp;  
  temp = fM[XT]; fM[XT] = -fM[TX]; fM[TX] = -temp;  
  temp = fM[YT]; fM[YT] = -fM[TY]; fM[TY] = -temp;  
  temp = fM[ZT]; fM[ZT] = -fM[TZ]; fM[TZ] = -temp;  
}

LorentzRotation
LorentzRotation::
Inverse() const {
  return LorentzRotation 
    (  fM[XX],  fM[YX],  fM[ZX], -fM[TX]
    ,  fM[XY],  fM[YY],  fM[ZY], -fM[TY]
    ,  fM[XZ],  fM[YZ],  fM[ZZ], -fM[TZ]
    , -fM[XT], -fM[YT], -fM[ZT],  fM[TT]
    );
}

LorentzRotation
LorentzRotation::
operator * (const LorentzRotation & r) const {
  return LorentzRotation 
    ( fM[XX]*r.fM[XX] + fM[XY]*r.fM[YX] + fM[XZ]*r.fM[ZX] + fM[XT]*r.fM[TX]
    , fM[XX]*r.fM[XY] + fM[XY]*r.fM[YY] + fM[XZ]*r.fM[ZY] + fM[XT]*r.fM[TY]
    , fM[XX]*r.fM[XZ] + fM[XY]*r.fM[YZ] + fM[XZ]*r.fM[ZZ] + fM[XT]*r.fM[TZ]
    , fM[XX]*r.fM[XT] + fM[XY]*r.fM[YT] + fM[XZ]*r.fM[ZT] + fM[XT]*r.fM[TT]
    , fM[YX]*r.fM[XX] + fM[YY]*r.fM[YX] + fM[YZ]*r.fM[ZX] + fM[YT]*r.fM[TX]
    , fM[YX]*r.fM[XY] + fM[YY]*r.fM[YY] + fM[YZ]*r.fM[ZY] + fM[YT]*r.fM[TY]
    , fM[YX]*r.fM[XZ] + fM[YY]*r.fM[YZ] + fM[YZ]*r.fM[ZZ] + fM[YT]*r.fM[TZ]
    , fM[YX]*r.fM[XT] + fM[YY]*r.fM[YT] + fM[YZ]*r.fM[ZT] + fM[YT]*r.fM[TT]
    , fM[ZX]*r.fM[XX] + fM[ZY]*r.fM[YX] + fM[ZZ]*r.fM[ZX] + fM[ZT]*r.fM[TX]
    , fM[ZX]*r.fM[XY] + fM[ZY]*r.fM[YY] + fM[ZZ]*r.fM[ZY] + fM[ZT]*r.fM[TY]
    , fM[ZX]*r.fM[XZ] + fM[ZY]*r.fM[YZ] + fM[ZZ]*r.fM[ZZ] + fM[ZT]*r.fM[TZ]
    , fM[ZX]*r.fM[XT] + fM[ZY]*r.fM[YT] + fM[ZZ]*r.fM[ZT] + fM[ZT]*r.fM[TT]
    , fM[TX]*r.fM[XX] + fM[TY]*r.fM[YX] + fM[TZ]*r.fM[ZX] + fM[TT]*r.fM[TX]
    , fM[TX]*r.fM[XY] + fM[TY]*r.fM[YY] + fM[TZ]*r.fM[ZY] + fM[TT]*r.fM[TY]
    , fM[TX]*r.fM[XZ] + fM[TY]*r.fM[YZ] + fM[TZ]*r.fM[ZZ] + fM[TT]*r.fM[TZ]
    , fM[TX]*r.fM[XT] + fM[TY]*r.fM[YT] + fM[TZ]*r.fM[ZT] + fM[TT]*r.fM[TT]
    );    
}


std::ostream & operator<< (std::ostream & os, const LorentzRotation & r) {
  // TODO - this will need changing for machine-readable issues
  //        and even the human readable form needs formatiing improvements
  double m[16];
  r.GetComponents(m, m+16);
  os << "\n" << m[0]  << "  " << m[1]  << "  " << m[2]  << "  " << m[3]; 
  os << "\n" << m[4]  << "  " << m[5]  << "  " << m[6]  << "  " << m[7]; 
  os << "\n" << m[8]  << "  " << m[9]  << "  " << m[10] << "  " << m[11]; 
  os << "\n" << m[12] << "  " << m[13] << "  " << m[14] << "  " << m[15] << "\n";
  return os;
}


} //namespace Math
} //namespace ROOT
