// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for rotation in 3 dimensions, represented by RotationZYX
//
// Created by: Lorenzo Moneta, May 23 2007
//
// Last update: $Id$
//
#include "MathX/GenVectorX/RotationZYX.h"

#include <cmath>

#include "MathX/GenVectorX/Cartesian3D.h"
#include "MathX/GenVectorX/DisplacementVector3D.h"
#include "MathX/GenVectorX/Rotation3D.h"
#include "MathX/GenVectorX/Quaternion.h"
#include "MathX/GenVectorX/RotationX.h"
#include "MathX/GenVectorX/RotationY.h"
#include "MathX/GenVectorX/RotationZ.h"

#include "MathX/GenVectorX/AxisAnglefwd.h"

#include "MathX/GenVectorX/AccHeaders.h"

#include "MathX/GenVectorX/MathHeaders.h"

namespace ROOT {

namespace ROOT_MATH_ARCH {

// ========== Constructors and Assignment =====================

// ========== Operations =====================

// DisplacementVector3D< Cartesian3D<double> >
// RotationZYX::
// operator() (const DisplacementVector3D< Cartesian3D<double> > & v) const
// {
//   return Rotation3D(*this)(v);
// }

RotationZYX RotationZYX::operator*(const Rotation3D &r) const
{
   // combine with a Rotation3D
   return RotationZYX(Rotation3D(*this) * r);
}

RotationZYX RotationZYX::operator*(const AxisAngle &a) const
{
   // combine with a AxisAngle
   return RotationZYX(Quaternion(*this) * Quaternion(a));
}

RotationZYX RotationZYX::operator*(const EulerAngles &e) const
{
   // combine with EulerAngles
   return RotationZYX(Quaternion(*this) * Quaternion(e));
}

RotationZYX RotationZYX::operator*(const RotationZYX &e) const
{
   // combine with a RotationZYX
   // return RotationZYX ( Quaternion(*this) * Quaternion(e) );
   return RotationZYX(Rotation3D(*this) * Rotation3D(e));
}
RotationZYX RotationZYX::operator*(const Quaternion &q) const
{
   // combination with a Quaternion
   return RotationZYX(Quaternion(*this) * q);
}

RotationZYX RotationZYX::operator*(const RotationX &r) const
{
   // combine with a RotationX
   return RotationZYX(Quaternion(*this) * r);
}

RotationZYX RotationZYX::operator*(const RotationY &r) const
{
   // combine with a RotationY
   return RotationZYX(Quaternion(*this) * r);
}

RotationZYX RotationZYX::operator*(const RotationZ &r) const
{
   // combine with a RotationZ
   // TODO -- this can be made much faster because it merely adds
   //         the r.Angle() to phi.
   Scalar newPhi = fPhi + r.Angle();
   if (newPhi <= -Pi() || newPhi > Pi()) {
      newPhi = newPhi - math_floor(newPhi / (2 * Pi()) + .5) * 2 * Pi();
   }
   return RotationZYX(newPhi, fTheta, fPsi);
}

RotationZYX operator*(RotationX const &r, RotationZYX const &e)
{
   return RotationZYX(r) * e; // TODO: improve performance
}

RotationZYX operator*(RotationY const &r, RotationZYX const &e)
{
   return RotationZYX(r) * e; // TODO: improve performance
}

RotationZYX operator*(RotationZ const &r, RotationZYX const &e)
{
   return RotationZYX(r) * e; // TODO: improve performance
}

void RotationZYX::Rectify()
{
   // rectify . The angle theta must be defined between [-PI/2,PI.2]
   //  same as Euler- Angles, just here Theta is shifted by PI/2 with respect to
   // the theta of the EulerAngles class

   Scalar theta2 = fTheta + M_PI_2;
   if (theta2 < 0 || theta2 > Pi()) {
      Scalar t = theta2 - math_floor(theta2 / (2 * Pi())) * 2 * Pi();
      if (t <= Pi()) {
         theta2 = t;
      } else {
         theta2 = 2 * Pi() - t;
         fPhi = fPhi + Pi();
         fPsi = fPsi + Pi();
      }
      // ftheta is shifted of PI/2 w.r.t theta2
      fTheta = theta2 - M_PI_2;
   }

   if (fPhi <= -Pi() || fPhi > Pi()) {
      fPhi = fPhi - math_floor(fPhi / (2 * Pi()) + .5) * 2 * Pi();
   }

   if (fPsi <= -Pi() || fPsi > Pi()) {
      fPsi = fPsi - math_floor(fPsi / (2 * Pi()) + .5) * 2 * Pi();
   }

} // Rectify()

void RotationZYX::Invert()
{
   // invert this rotation.
   // use Rotation3D. TO Do :have algorithm to invert it directly
   Rotation3D r(*this);
   // Quaternion r(*this);
   r.Invert();
   *this = r;
}

#if !defined(ROOT_MATH_SYCL) && !defined(ROOT_MATH_CUDA)

// ========== I/O =====================

std::ostream &operator<<(std::ostream &os, const RotationZYX &e)
{
   // TODO - this will need changing for machine-readable issues
   //        and even the human readable form may need formatting improvements
   os << "\n{phi(Z angle): " << e.Phi() << "   theta(Y angle): " << e.Theta() << "   psi(X angle): " << e.Psi()
      << "}\n";
   return os;
}
#endif

} // namespace ROOT_MATH_ARCH
} // namespace ROOT
