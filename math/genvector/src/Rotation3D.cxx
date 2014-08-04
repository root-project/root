// @(#)root/mathcore:$Id$
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

Rotation3D::Rotation3D()
{
   // constructor of a identity rotation
   fM[kXX] = 1.0;  fM[kXY] = 0.0; fM[kXZ] = 0.0;
   fM[kYX] = 0.0;  fM[kYY] = 1.0; fM[kYZ] = 0.0;
   fM[kZX] = 0.0;  fM[kZY] = 0.0; fM[kZZ] = 1.0;
}


void Rotation3D::Rectify()
{
   // rectify rotation matrix (make orthogonal)
   // The "nearest" orthogonal matrix X to a nearly-orthogonal matrix A
   // (in the sense that X is exaclty orthogonal and the sum of the squares
   // of the element differences X-A is as small as possible) is given by
   // X = A * inverse(sqrt(A.transpose()*A.inverse())).

   // Step 1 -- form symmetric M = A.transpose * A

   double m11 = fM[kXX]*fM[kXX] + fM[kYX]*fM[kYX] + fM[kZX]*fM[kZX];
   double m12 = fM[kXX]*fM[kXY] + fM[kYX]*fM[kYY] + fM[kZX]*fM[kZY];
   double m13 = fM[kXX]*fM[kXZ] + fM[kYX]*fM[kYZ] + fM[kZX]*fM[kZZ];
   double m22 = fM[kXY]*fM[kXY] + fM[kYY]*fM[kYY] + fM[kZY]*fM[kZY];
   double m23 = fM[kXY]*fM[kXZ] + fM[kYY]*fM[kYZ] + fM[kZY]*fM[kZZ];
   double m33 = fM[kXZ]*fM[kXZ] + fM[kYZ]*fM[kYZ] + fM[kZZ]*fM[kZZ];

   // Step 2 -- find lower-triangular U such that U * U.transpose = M

   double u11 = std::sqrt(m11);
   double u21 = m12/u11;
   double u31 = m13/u11;
   double u22 = std::sqrt(m22-u21*u21);
   double u32 = (m23-m12*m13/m11)/u22;
   double u33 = std::sqrt(m33 - u31*u31 - u32*u32);


   // Step 3 -- find V such that V*V = U.  U is also lower-triangular

   double v33 = 1/u33;
   double v32 = -v33*u32/u22;
   double v31 = -(v32*u21+v33*u31)/u11;
   double v22 = 1/u22;
   double v21 = -v22*u21/u11;
   double v11 = 1/u11;


   // Step 4 -- N = V.transpose * V is inverse(sqrt(A.transpose()*A.inverse()))

   double n11 = v11*v11 + v21*v21 + v31*v31;
   double n12 = v11*v21 + v21*v22 + v31*v32;
   double n13 = v11*v31 + v21*v32 + v31*v33;
   double n22 = v21*v21 + v22*v22 + v32*v32;
   double n23 = v21*v31 + v22*v32 + v32*v33;
   double n33 = v31*v31 + v32*v32 + v33*v33;


   // Step 5 -- The new matrix is A * N

   double mA[9];
   std::copy(fM, &fM[9], mA);

   fM[kXX] = mA[kXX]*n11 + mA[kXY]*n12 + mA[kXZ]*n13;
   fM[kXY] = mA[kXX]*n12 + mA[kXY]*n22 + mA[kXZ]*n23;
   fM[kXZ] = mA[kXX]*n13 + mA[kXY]*n23 + mA[kXZ]*n33;
   fM[kYX] = mA[kYX]*n11 + mA[kYY]*n12 + mA[kYZ]*n13;
   fM[kYY] = mA[kYX]*n12 + mA[kYY]*n22 + mA[kYZ]*n23;
   fM[kYZ] = mA[kYX]*n13 + mA[kYY]*n23 + mA[kYZ]*n33;
   fM[kZX] = mA[kZX]*n11 + mA[kZY]*n12 + mA[kZZ]*n13;
   fM[kZY] = mA[kZX]*n12 + mA[kZY]*n22 + mA[kZZ]*n23;
   fM[kZZ] = mA[kZX]*n13 + mA[kZY]*n23 + mA[kZZ]*n33;


} // Rectify()


static inline void swap(double & a, double & b) {
   // swap two values
   double t=b; b=a; a=t;
}

void Rotation3D::Invert() {
   // invert a rotation
   swap (fM[kXY], fM[kYX]);
   swap (fM[kXZ], fM[kZX]);
   swap (fM[kYZ], fM[kZY]);
}


Rotation3D Rotation3D::operator * (const AxisAngle   & a) const {
   // combine with an AxisAngle rotation
   return operator* ( Rotation3D(a) );
}

Rotation3D Rotation3D::operator * (const EulerAngles & e) const {
   // combine with an EulerAngles rotation
   return operator* ( Rotation3D(e) );
}

Rotation3D Rotation3D::operator * (const Quaternion  & q) const {
   // combine with a Quaternion rotation
   return operator* ( Rotation3D(q) );
}

Rotation3D Rotation3D::operator * (const RotationZYX  & r) const {
   // combine with a RotastionZYX rotation
   return operator* ( Rotation3D(r) );
}

std::ostream & operator<< (std::ostream & os, const Rotation3D & r) {
   // TODO - this will need changing for machine-readable issues
   //        and even the human readable form needs formatting improvements
   double m[9];
   r.GetComponents(m, m+9);
   os << "\n" << m[0] << "  " << m[1] << "  " << m[2];
   os << "\n" << m[3] << "  " << m[4] << "  " << m[5];
   os << "\n" << m[6] << "  " << m[7] << "  " << m[8] << "\n";
   return os;
}

} //namespace Math
} //namespace ROOT
