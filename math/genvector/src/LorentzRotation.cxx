// @(#)root/mathcore:$Id$
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
   // constructor of an identity LR
   fM[kXX] = 1.0;  fM[kXY] = 0.0; fM[kXZ] = 0.0; fM[kXT] = 0.0;
   fM[kYX] = 0.0;  fM[kYY] = 1.0; fM[kYZ] = 0.0; fM[kYT] = 0.0;
   fM[kZX] = 0.0;  fM[kZY] = 0.0; fM[kZZ] = 1.0; fM[kZT] = 0.0;
   fM[kTX] = 0.0;  fM[kTY] = 0.0; fM[kTZ] = 0.0; fM[kTT] = 1.0;
}

LorentzRotation::LorentzRotation(Rotation3D  const & r) {
   // construct from  Rotation3D
   r.GetComponents ( fM[kXX], fM[kXY], fM[kXZ],
                     fM[kYX], fM[kYY], fM[kYZ],
                     fM[kZX], fM[kZY], fM[kZZ] );
   fM[kXT] = 0.0;
   fM[kYT] = 0.0;
   fM[kZT] = 0.0;
   fM[kTX] = 0.0;  fM[kTY] = 0.0; fM[kTZ] = 0.0; fM[kTT] = 1.0;
}

LorentzRotation::LorentzRotation(AxisAngle  const & a) {
   // construct from  AxisAngle
   const Rotation3D r(a);
   r.GetComponents ( fM[kXX], fM[kXY], fM[kXZ],
                     fM[kYX], fM[kYY], fM[kYZ],
                     fM[kZX], fM[kZY], fM[kZZ] );
   fM[kXT] = 0.0;
   fM[kYT] = 0.0;
   fM[kZT] = 0.0;
   fM[kTX] = 0.0;  fM[kTY] = 0.0; fM[kTZ] = 0.0; fM[kTT] = 1.0;
}

LorentzRotation::LorentzRotation(EulerAngles  const & e) {
   // construct from  EulerAngles
   const Rotation3D r(e);
   r.GetComponents ( fM[kXX], fM[kXY], fM[kXZ],
                     fM[kYX], fM[kYY], fM[kYZ],
                     fM[kZX], fM[kZY], fM[kZZ] );
   fM[kXT] = 0.0;
   fM[kYT] = 0.0;
   fM[kZT] = 0.0;
   fM[kTX] = 0.0;  fM[kTY] = 0.0; fM[kTZ] = 0.0; fM[kTT] = 1.0;
}

LorentzRotation::LorentzRotation(Quaternion  const & q) {
   // construct from Quaternion
   const Rotation3D r(q);
   r.GetComponents ( fM[kXX], fM[kXY], fM[kXZ],
                     fM[kYX], fM[kYY], fM[kYZ],
                     fM[kZX], fM[kZY], fM[kZZ] );
   fM[kXT] = 0.0;
   fM[kYT] = 0.0;
   fM[kZT] = 0.0;
   fM[kTX] = 0.0;  fM[kTY] = 0.0; fM[kTZ] = 0.0; fM[kTT] = 1.0;
}

LorentzRotation::LorentzRotation(RotationX  const & r) {
   // construct from  RotationX
   Scalar s = r.SinAngle();
   Scalar c = r.CosAngle();
   fM[kXX] = 1.0;  fM[kXY] = 0.0; fM[kXZ] = 0.0; fM[kXT] = 0.0;
   fM[kYX] = 0.0;  fM[kYY] =  c ; fM[kYZ] = -s ; fM[kYT] = 0.0;
   fM[kZX] = 0.0;  fM[kZY] =  s ; fM[kZZ] =  c ; fM[kZT] = 0.0;
   fM[kTX] = 0.0;  fM[kTY] = 0.0; fM[kTZ] = 0.0; fM[kTT] = 1.0;
}

LorentzRotation::LorentzRotation(RotationY  const & r) {
   // construct from  RotationY
   Scalar s = r.SinAngle();
   Scalar c = r.CosAngle();
   fM[kXX] =  c ;  fM[kXY] = 0.0; fM[kXZ] =  s ; fM[kXT] = 0.0;
   fM[kYX] = 0.0;  fM[kYY] = 1.0; fM[kYZ] = 0.0; fM[kYT] = 0.0;
   fM[kZX] = -s ;  fM[kZY] = 0.0; fM[kZZ] =  c ; fM[kZT] = 0.0;
   fM[kTX] = 0.0;  fM[kTY] = 0.0; fM[kTZ] = 0.0; fM[kTT] = 1.0;
}

LorentzRotation::LorentzRotation(RotationZ  const & r) {
   // construct from  RotationX
   Scalar s = r.SinAngle();
   Scalar c = r.CosAngle();
   fM[kXX] =  c ;  fM[kXY] = -s ; fM[kXZ] = 0.0; fM[kXT] = 0.0;
   fM[kYX] =  s ;  fM[kYY] =  c ; fM[kYZ] = 0.0; fM[kYT] = 0.0;
   fM[kZX] = 0.0;  fM[kZY] = 0.0; fM[kZZ] = 1.0; fM[kZT] = 0.0;
   fM[kTX] = 0.0;  fM[kTY] = 0.0; fM[kTZ] = 0.0; fM[kTT] = 1.0;
}

void
LorentzRotation::Rectify() {
   // Assuming the representation of this is close to a true Lorentz Rotation,
   // but may have drifted due to round-off error from many operations,
   // this forms an "exact" orthosymplectic matrix for the Lorentz Rotation
   // again.

   typedef LorentzVector< PxPyPzE4D<Scalar> > FourVector;
   if (fM[kTT] <= 0) {
      GenVector::Throw (
                              "LorentzRotation:Rectify(): Non-positive TT component - cannot rectify");
      return;
   }
   FourVector t ( fM[kTX], fM[kTY], fM[kTZ], fM[kTT] );
   Scalar m2 = t.M2();
   if ( m2 <= 0 ) {
      GenVector::Throw (
                              "LorentzRotation:Rectify(): Non-timelike time row - cannot rectify");
      return;
   }
   t /= std::sqrt(m2);
   FourVector z ( fM[kZX], fM[kZY], fM[kZZ], fM[kZT] );
   z = z - z.Dot(t)*t;
   m2 = z.M2();
   if ( m2 >= 0 ) {
      GenVector::Throw (
                              "LorentzRotation:Rectify(): Non-spacelike Z row projection - "
                              "cannot rectify");
      return;
   }
   z /= std::sqrt(-m2);
   FourVector y ( fM[kYX], fM[kYY], fM[kYZ], fM[kYT] );
   y = y - y.Dot(t)*t - y.Dot(z)*z;
   m2 = y.M2();
   if ( m2 >= 0 ) {
      GenVector::Throw (
                              "LorentzRotation:Rectify(): Non-spacelike Y row projection - "
                              "cannot rectify");
      return;
   }
   y /= std::sqrt(-m2);
   FourVector x ( fM[kXX], fM[kXY], fM[kXZ], fM[kXT] );
   x = x - x.Dot(t)*t - x.Dot(z)*z - x.Dot(y)*y;
   m2 = x.M2();
   if ( m2 >= 0 ) {
      GenVector::Throw (
                              "LorentzRotation:Rectify(): Non-spacelike X row projection - "
                              "cannot rectify");
      return;
   }
   x /= std::sqrt(-m2);
}


void LorentzRotation::Invert() {
   // invert modifying current content
   Scalar temp;
   temp = fM[kXY]; fM[kXY] =  fM[kYX]; fM[kYX] =  temp;
   temp = fM[kXZ]; fM[kXZ] =  fM[kZX]; fM[kZX] =  temp;
   temp = fM[kYZ]; fM[kYZ] =  fM[kZY]; fM[kZY] =  temp;
   temp = fM[kXT]; fM[kXT] = -fM[kTX]; fM[kTX] = -temp;
   temp = fM[kYT]; fM[kYT] = -fM[kTY]; fM[kTY] = -temp;
   temp = fM[kZT]; fM[kZT] = -fM[kTZ]; fM[kTZ] = -temp;
}

LorentzRotation LorentzRotation::Inverse() const {
   // return an inverse LR
   return LorentzRotation
   (  fM[kXX],  fM[kYX],  fM[kZX], -fM[kTX]
      ,  fM[kXY],  fM[kYY],  fM[kZY], -fM[kTY]
      ,  fM[kXZ],  fM[kYZ],  fM[kZZ], -fM[kTZ]
      , -fM[kXT], -fM[kYT], -fM[kZT],  fM[kTT]
      );
}

LorentzRotation LorentzRotation::operator * (const LorentzRotation & r) const {
   // combination with another LR
   return LorentzRotation
   ( fM[kXX]*r.fM[kXX] + fM[kXY]*r.fM[kYX] + fM[kXZ]*r.fM[kZX] + fM[kXT]*r.fM[kTX]
     , fM[kXX]*r.fM[kXY] + fM[kXY]*r.fM[kYY] + fM[kXZ]*r.fM[kZY] + fM[kXT]*r.fM[kTY]
     , fM[kXX]*r.fM[kXZ] + fM[kXY]*r.fM[kYZ] + fM[kXZ]*r.fM[kZZ] + fM[kXT]*r.fM[kTZ]
     , fM[kXX]*r.fM[kXT] + fM[kXY]*r.fM[kYT] + fM[kXZ]*r.fM[kZT] + fM[kXT]*r.fM[kTT]
     , fM[kYX]*r.fM[kXX] + fM[kYY]*r.fM[kYX] + fM[kYZ]*r.fM[kZX] + fM[kYT]*r.fM[kTX]
     , fM[kYX]*r.fM[kXY] + fM[kYY]*r.fM[kYY] + fM[kYZ]*r.fM[kZY] + fM[kYT]*r.fM[kTY]
     , fM[kYX]*r.fM[kXZ] + fM[kYY]*r.fM[kYZ] + fM[kYZ]*r.fM[kZZ] + fM[kYT]*r.fM[kTZ]
     , fM[kYX]*r.fM[kXT] + fM[kYY]*r.fM[kYT] + fM[kYZ]*r.fM[kZT] + fM[kYT]*r.fM[kTT]
     , fM[kZX]*r.fM[kXX] + fM[kZY]*r.fM[kYX] + fM[kZZ]*r.fM[kZX] + fM[kZT]*r.fM[kTX]
     , fM[kZX]*r.fM[kXY] + fM[kZY]*r.fM[kYY] + fM[kZZ]*r.fM[kZY] + fM[kZT]*r.fM[kTY]
     , fM[kZX]*r.fM[kXZ] + fM[kZY]*r.fM[kYZ] + fM[kZZ]*r.fM[kZZ] + fM[kZT]*r.fM[kTZ]
     , fM[kZX]*r.fM[kXT] + fM[kZY]*r.fM[kYT] + fM[kZZ]*r.fM[kZT] + fM[kZT]*r.fM[kTT]
     , fM[kTX]*r.fM[kXX] + fM[kTY]*r.fM[kYX] + fM[kTZ]*r.fM[kZX] + fM[kTT]*r.fM[kTX]
     , fM[kTX]*r.fM[kXY] + fM[kTY]*r.fM[kYY] + fM[kTZ]*r.fM[kZY] + fM[kTT]*r.fM[kTY]
     , fM[kTX]*r.fM[kXZ] + fM[kTY]*r.fM[kYZ] + fM[kTZ]*r.fM[kZZ] + fM[kTT]*r.fM[kTZ]
     , fM[kTX]*r.fM[kXT] + fM[kTY]*r.fM[kYT] + fM[kTZ]*r.fM[kZT] + fM[kTT]*r.fM[kTT]
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
