// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for rotation in 3 dimensions, represented by quaternion
//
// Created by: Mark Fischler Thurs June 9  2005
//
// Last update: $Id$
//
#include "MathX/GenVectorX/Quaternion.h"

#include <cmath>

#include "MathX/GenVectorX/Cartesian3D.h"
#include "MathX/GenVectorX/DisplacementVector3D.h"

#include "MathX/GenVectorX/Rotation3Dfwd.h"
#include "MathX/GenVectorX/AxisAnglefwd.h"
#include "MathX/GenVectorX/EulerAnglesfwd.h"

#include "MathX/GenVectorX/AccHeaders.h"

#include "MathX/GenVectorX/MathHeaders.h"

namespace ROOT {

namespace ROOT_MATH_ARCH {

// ========== Constructors and Assignment =====================

void Quaternion::Rectify()
{

   // The vector should be a unit vector, and the first element should be
   // non-negative (though negative fU quaternions would work just fine,
   // being isomorphic to a quaternion with positive fU).

   if (fU < 0) {
      fU = -fU;
      fI = -fI;
      fJ = -fJ;
      fK = -fK;
   }

   Scalar a = 1.0 / math_sqrt(fU * fU + fI * fI + fJ * fJ + fK * fK);
   fU *= a;
   fI *= a;
   fJ *= a;
   fK *= a;

} // Rectify()

// ========== Operations =====================

// DisplacementVector3D< Cartesian3D<double> >
// Quaternion::operator() (const DisplacementVector3D< Cartesian3D<double> > & v) const
// {
//    // apply to a 3D Vector
// }

// Quaternion Quaternion::operator * (const Quaternion & q) const {
//    // combination of rotations
//    return Quaternion                          (
//                                                fU*q.fU - fI*q.fI - fJ*q.fJ - fK*q.fK
//                                                , fU*q.fI + fI*q.fU + fJ*q.fK - fK*q.fJ
//                                                , fU*q.fJ - fI*q.fK + fJ*q.fU + fK*q.fI
//                                                , fU*q.fK + fI*q.fJ - fJ*q.fI + fK*q.fU  );
//}

Quaternion Quaternion::operator*(const Rotation3D &r) const
{
   // combination of rotations
   return operator*(Quaternion(r));
}

Quaternion Quaternion::operator*(const AxisAngle &a) const
{
   // combination of rotations
   return operator*(Quaternion(a));
}

Quaternion Quaternion::operator*(const EulerAngles &e) const
{
   // combination of rotations
   return operator*(Quaternion(e));
}

Quaternion Quaternion::operator*(const RotationZYX &r) const
{
   // combination of rotations
   return operator*(Quaternion(r));
}

Quaternion::Scalar Quaternion::Distance(const Quaternion &q) const
{
   // distance
   Scalar chordLength = math_fabs(fU * q.fU + fI * q.fI + fJ * q.fJ + fK * q.fK);
   if (chordLength > 1)
      chordLength = 1; // in case roundoff fouls us up
   return acos(chordLength);
}

#if !defined(ROOT_MATH_SYCL) && !defined(ROOT_MATH_CUDA)
// ========== I/O =====================

std::ostream &operator<<(std::ostream &os, const Quaternion &q)
{
   // TODO - this will need changing for machine-readable issues
   //        and even the human readable form may need formatting improvements
   os << "\n{" << q.U() << "   " << q.I() << "   " << q.J() << "   " << q.K() << "}\n";
   return os;
}
#endif

} // namespace ROOT_MATH_ARCH
} // namespace ROOT
