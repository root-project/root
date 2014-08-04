// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// implementation file for class Translation3D
//
// Created by: Lorenzo Moneta  October 27 2005
//
//

#include "Math/GenVector/Translation3D.h"
#include "Math/GenVector/Plane3D.h"
#include "Math/GenVector/PositionVector3D.h"

#include <cmath>
#include <algorithm>




namespace ROOT {

namespace Math {


typedef Translation3D::Vector XYZVector;
typedef PositionVector3D<Cartesian3D<double> > XYZPoint;


// ========== Constructors and Assignment =====================


Plane3D Translation3D::operator() (const Plane3D & plane) const
{
   // transformations on a 3D plane
   XYZVector n = plane.Normal();
   // take a point on the plane. Use origin projection on the plane
   // ( -ad, -bd, -cd) if (a**2 + b**2 + c**2 ) = 1
   double d = plane.HesseDistance();
   XYZPoint p( - d * n.X() , - d *n.Y(), -d *n.Z() );
   return Plane3D ( operator() (n), operator() (p) );
}

std::ostream & operator<< (std::ostream & os, const Translation3D & t)
{
   // TODO - this will need changing for machine-readable issues
   //        and even the human readable form needs formatiing improvements

   double m[3];
   t.GetComponents(m, m+3);
   os << "\n" << m[0] << "  " << m[1] << "  " << m[2] << "\n";
   return os;
}

}  // end namespace Math
}  // end namespace ROOT
