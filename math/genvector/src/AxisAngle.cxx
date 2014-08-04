// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class AxisAngle, a rotation in 3 dimensions
// represented by its axis and angle of rotation
//
// Created by: Mark Fischler Tues July 5  2005
//
#include "Math/GenVector/AxisAngle.h"

#include <cmath>
#include <algorithm>

#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Rotation3D.h"

namespace ROOT {

namespace Math {

// ========== Constructors and Assignment =====================

void AxisAngle::RectifyAngle() {
   // Note: We could require the angle to be in [0,pi) since we
   //       can represent negative angles by flipping the axis.
   //       We choose not to do this.

   if ( fAngle <= Pi() && fAngle > -Pi() ) return;

   if ( fAngle > 0 ) {
      int n = static_cast<int>( (fAngle+Pi())/(2*Pi()) );
      fAngle -= 2*Pi()*n;
   } else {
      int n = static_cast<int>( -(fAngle-Pi())/(2*Pi()) );
      fAngle += 2*Pi()*n;
   }
} // RectifyAngle()

void AxisAngle::Rectify()
{
   // The two conditions are that the angle is in (-pi, pi] and
   // the axis is a unit vector.

   Scalar r2 = fAxis.Mag2();
   if ( r2 == 0 ) {
      fAxis.SetCoordinates(0,0,1);
      fAngle = 0;
      return;
   }
   fAxis *= (1.0/r2);
   RectifyAngle();
} // Rectify()

// ======== Transformation to other Rotation Forms ==================

enum ERotation3DMatrixIndex {
   kXX = 0, kXY = 1, kXZ = 2
   , kYX = 3, kYY = 4, kYZ = 5
   , kZX = 6, kZY = 7, kZZ = 8
};



// ========== Operations =====================

DisplacementVector3D< Cartesian3D<double> >
AxisAngle::
operator() (const DisplacementVector3D< Cartesian3D<double> > & v) const
{
   Scalar c = std::cos(fAngle);
   Scalar s = std::sin(fAngle);
   Scalar p = fAxis.Dot(v) * ( 1 - c );
   return  DisplacementVector3D< Cartesian3D<double> >
      (
       c*v.X() + p*fAxis.X() + s * (fAxis.Y()*v.Z() - fAxis.Z()*v.Y())
       , c*v.Y() + p*fAxis.Y() + s * (fAxis.Z()*v.X() - fAxis.X()*v.Z())
       , c*v.Z() + p*fAxis.Z() + s * (fAxis.X()*v.Y() - fAxis.Y()*v.X())
       );
}

// ========== I/O =====================

std::ostream & operator<< (std::ostream & os, const AxisAngle & a) {
   // TODO - this will need changing for machine-readable issues
   //        and even the human readable form may need formatiing improvements
   os << "\n" << a.Axis() << "  " << a.Angle() << "\n";
   return os;
}



} //namespace Math
} //namespace ROOT
