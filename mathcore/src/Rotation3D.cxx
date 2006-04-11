// @(#)root/mathcore:$Name:  $:$Id: Rotation3D.cxx,v 1.6 2006/02/26 17:20:11 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Rotation in 3 dimensions, represented by 3x3 matrix
//
// Created by: Mark Fischler Tues July 5 2005
//
#include "Math/GenVector/Rotation3D.h"

#include <cmath>
#include <algorithm>

#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"

namespace ROOT {

  namespace Math {

// ========== Constructors and Assignment =====================

Rotation3D::Rotation3D() {
  fM[XX] = 1.0;  fM[XY] = 0.0; fM[XZ] = 0.0;
  fM[YX] = 0.0;  fM[YY] = 1.0; fM[YZ] = 0.0;
  fM[ZX] = 0.0;  fM[ZY] = 0.0; fM[ZZ] = 1.0;
}
 

void
Rotation3D::Rectify()
{

  // The "nearest" orthogonal matrix X to a nearly-orthogonal matrix A
  // (in the sense that X is exaclty orthogonal and the sum of the squares
  // of the element differences X-A is as small as possible) is given by
  // X = A * inverse(sqrt(A.transpose()*A.inverse())).

  // Step 1 -- form symmetric M = A.transpose * A

  double M11 = fM[XX]*fM[XX] + fM[YX]*fM[YX] + fM[ZX]*fM[ZX];
  double M12 = fM[XX]*fM[XY] + fM[YX]*fM[YY] + fM[ZX]*fM[ZY];
  double M13 = fM[XX]*fM[XZ] + fM[YX]*fM[YZ] + fM[ZX]*fM[ZZ];
  double M22 = fM[XY]*fM[XY] + fM[YY]*fM[YY] + fM[ZY]*fM[ZY];
  double M23 = fM[XY]*fM[XZ] + fM[YY]*fM[YZ] + fM[ZY]*fM[ZZ];
  double M33 = fM[XZ]*fM[XZ] + fM[YZ]*fM[YZ] + fM[ZZ]*fM[ZZ];

  // Step 2 -- find lower-triangular L such that L * L.transpose = M

  double L11 = std::sqrt(M11);
  double L21 = M12/L11;
  double L31 = M13/L11;
  double L22 = std::sqrt(M22-L21*L21);
  double L32 = (M23-M12*M13/M11)/L22;
  double L33 = std::sqrt(M33 - L31*L31 - L32*L32);


  // Step 3 -- find K such that K*K = L.  K is also lower-triangular

  double K33 = 1/L33;
  double K32 = -K33*L32/L22;
  double K31 = -(K32*L21+K33*L31)/L11;
  double K22 = 1/L22;
  double K21 = -K22*L21/L11;
  double K11 = 1/L11;


  // Step 4 -- N = K.transpose * K is inverse(sqrt(A.transpose()*A.inverse()))

  double N11 = K11*K11 + K21*K21 + K31*K31;
  double N12 = K11*K21 + K21*K22 + K31*K32;
  double N13 = K11*K31 + K21*K32 + K31*K33;
  double N22 = K21*K21 + K22*K22 + K32*K32;
  double N23 = K21*K31 + K22*K32 + K32*K33;
  double N33 = K31*K31 + K32*K32 + K33*K33;


  // Step 5 -- The new matrix is A * N

  double A[9];
  std::copy(fM, &fM[9], A);

  fM[XX] = A[XX]*N11 + A[XY]*N12 + A[XZ]*N13;
  fM[XY] = A[XX]*N12 + A[XY]*N22 + A[XZ]*N23;
  fM[XZ] = A[XX]*N13 + A[XY]*N23 + A[XZ]*N33;
  fM[YX] = A[YX]*N11 + A[YY]*N12 + A[YZ]*N13;
  fM[YY] = A[YX]*N12 + A[YY]*N22 + A[YZ]*N23;
  fM[YZ] = A[YX]*N13 + A[YY]*N23 + A[YZ]*N33;
  fM[ZX] = A[ZX]*N11 + A[ZY]*N12 + A[ZZ]*N13;
  fM[ZY] = A[ZX]*N12 + A[ZY]*N22 + A[ZZ]*N23;
  fM[ZZ] = A[ZX]*N13 + A[ZY]*N23 + A[ZZ]*N33;


} // Rectify()


// ========== Operations =====================

// DisplacementVector3D< Cartesian3D<double> >
// Rotation3D::
// operator() (const DisplacementVector3D< Cartesian3D<double> > & v) const
// {
//   return  DisplacementVector3D< Cartesian3D<double> >  (
//       fM[XX] * v.X() + fM[XY] * v.Y() + fM[XZ] * v.Z()
//     , fM[YX] * v.X() + fM[YY] * v.Y() + fM[YZ] * v.Z()
//     , fM[ZX] * v.X() + fM[ZY] * v.Y() + fM[ZZ] * v.Z() );
// }

static inline void swap(double & a, double & b) { double t=b; b=a; a=t; }

void
Rotation3D::
Invert() {
  swap (fM[XY], fM[YX]);
  swap (fM[XZ], fM[ZX]);
  swap (fM[YZ], fM[ZY]);
}

Rotation3D
Rotation3D::
operator * (const Rotation3D  & r) const {
  return Rotation3D 
  (
    fM[XX]*r.fM[XX] + fM[XY]*r.fM[YX] + fM[XZ]*r.fM[ZX]
  , fM[XX]*r.fM[XY] + fM[XY]*r.fM[YY] + fM[XZ]*r.fM[ZY]
  , fM[XX]*r.fM[XZ] + fM[XY]*r.fM[YZ] + fM[XZ]*r.fM[ZZ]
  
  , fM[YX]*r.fM[XX] + fM[YY]*r.fM[YX] + fM[YZ]*r.fM[ZX]
  , fM[YX]*r.fM[XY] + fM[YY]*r.fM[YY] + fM[YZ]*r.fM[ZY]
  , fM[YX]*r.fM[XZ] + fM[YY]*r.fM[YZ] + fM[YZ]*r.fM[ZZ]
  
  , fM[ZX]*r.fM[XX] + fM[ZY]*r.fM[YX] + fM[ZZ]*r.fM[ZX]
  , fM[ZX]*r.fM[XY] + fM[ZY]*r.fM[YY] + fM[ZZ]*r.fM[ZY]
  , fM[ZX]*r.fM[XZ] + fM[ZY]*r.fM[YZ] + fM[ZZ]*r.fM[ZZ]
  );
}

Rotation3D
Rotation3D::
operator * (const AxisAngle   & a) const {
  return operator* ( Rotation3D(a) );
}

Rotation3D
Rotation3D::
operator * (const EulerAngles & e) const {
  return operator* ( Rotation3D(e) );
}

Rotation3D
Rotation3D::
operator * (const Quaternion  & q) const {
  return operator* ( Rotation3D(q) );
}

std::ostream & operator<< (std::ostream & os, const Rotation3D & r) {
  // TODO - this will need changing for machine-readable issues
  //        and even the human readable form needs formatiing improvements
  double m[9];
  r.GetComponents(m, m+9);
  os << "\n" << m[0] << "  " << m[1] << "  " << m[2]; 
  os << "\n" << m[3] << "  " << m[4] << "  " << m[5]; 
  os << "\n" << m[6] << "  " << m[7] << "  " << m[8] << "\n";
  return os;
}

} //namespace Math
} //namespace ROOT
