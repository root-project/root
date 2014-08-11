// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// implementation file for class Plane3D
//
// Created by: Lorenzo Moneta  December 2 2005
//
//

#include "Math/GenVector/Plane3D.h"

#include <cmath>




namespace ROOT {

namespace Math {


typedef Plane3D::Scalar Scalar;
typedef Plane3D::Point  XYZPoint;
typedef Plane3D::Vector XYZVector;

// ========== Constructors and Assignment =====================


// constructor from 4 scalars numbers (a,b,c,d)
Plane3D::Plane3D(const Scalar & a, const Scalar & b, const Scalar & c, const Scalar & d) :
   fA(a), fB(b), fC(c), fD(d)
{
   //renormalize a,b,c to unit
   Normalize();
}

// internal method to construct from a normal vector and a point
void Plane3D::BuildFromVecAndPoint(const XYZVector & n, const XYZPoint & p )
{
   // build from a normal vector and a point
   fA =  n.X();
   fB =  n.Y();
   fC =  n.Z();
   fD = - n.Dot(p);
   Normalize();
}

// internl method to construct from three points
void Plane3D::BuildFrom3Points( const XYZPoint & p1, const XYZPoint & p2, const XYZPoint & p3 ) {

   // plane from thre points
   // normal is (x3-x1) cross (x2 -x1)
   XYZVector n = (p2-p1).Cross(p3-p1);
   fA = n.X();
   fB = n.Y();
   fC = n.Z();
   fD = - n.Dot(p1);
   Normalize();
}

// distance plane- point
Scalar Plane3D::Distance(const XYZPoint & p) const {
   return fA*p.X() + fB*p.Y() + fC*p.Z() + fD;
}

void Plane3D::Normalize() {
   // normalize the plane
   Scalar s = std::sqrt( fA*fA + fB*fB + fC*fC );
   // what to do if s = 0 ??
   if ( s == 0) { fD = 0; return; }
   Scalar w = 1./s;
   fA *= w;
   fB *= w;
   fC *= w;
   fD *= w;
}


// projection of a point onto the plane
XYZPoint Plane3D::ProjectOntoPlane(const XYZPoint & p) const {
   Scalar d = Distance(p);
   return XYZPoint( p.X() - fA*d, p.Y() - fB*d, p.Z() - fC*d);
}


// output
std::ostream & operator<< (std::ostream & os, const Plane3D & p) {
   os << "\n" << p.Normal().X()
   << "  " << p.Normal().Y()
   << "  " << p.Normal().Z()
   << "  " << p.HesseDistance()
   << "\n";
   return os;
}





}  // end namespace Math
}  // end namespace ROOT

