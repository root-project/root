// @(#)root/mathcore:$Id$
// Authors: L. Moneta    12/2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class LorentzVector
//
// Created by:    moneta   at Fri Dec 02   2005
//
// Last update: $Id$
//
#ifndef ROOT_Math_GenVector_Plane3D
#define ROOT_Math_GenVector_Plane3D  1

#include <type_traits>

#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/PositionVector3D.h"



namespace ROOT {

namespace Math {

namespace Impl {

//_______________________________________________________________________________
/**
   Class describing a geometrical plane in 3 dimensions.
   A Plane3D is a 2 dimensional surface spanned by two linearly independent vectors.
   The plane is described by the equation
   \f$ a*x + b*y + c*z + d = 0 \f$ where (a,b,c) are the components of the
   normal vector to the plane \f$ n = (a,b,c)  \f$ and \f$ d = - n \dot x \f$, where x is any point
   belonging to plane.
   More information on the mathematics describing a plane in 3D is available on
   <A HREF=http://mathworld.wolfram.com/Plane.html>MathWord</A>.
   The Plane3D class contains the 4 scalar values in T which represent the
   four coefficients, fA, fB, fC, fD. fA, fB, fC are the normal components normalized to 1,
   i.e. fA**2 + fB**2 + fC**2 = 1

   @ingroup GenVector
*/

template <typename T = double>
class Plane3D {

public:
   // ------ ctors ------

   typedef T Scalar;

   typedef DisplacementVector3D<Cartesian3D<T>, DefaultCoordinateSystemTag> Vector;
   typedef PositionVector3D<Cartesian3D<T>, DefaultCoordinateSystemTag>     Point;

   /**
      default constructor create plane z = 0
   */
   Plane3D() : fA(0), fB(0), fC(1), fD(0) {}

   /**
    generic constructors from the four scalar values describing the plane
    according to the equation ax + by + cz + d = 0
      \param a scalar value
      \param b scalar value
      \param c scalar value
      \param d sxcalar value
   */
   Plane3D(const Scalar &a, const Scalar &b, const Scalar &c, const Scalar &d) : fA(a), fB(b), fC(c), fD(d)
   {
      // renormalize a,b,c to unit
      Normalize();
   }

   /**
    constructor a Plane3D from a normal vector and a point coplanar to the plane
    \param n normal expressed as a ROOT::Math::DisplacementVector3D<Cartesian3D<T> >
    \param p point  expressed as a  ROOT::Math::PositionVector3D<Cartesian3D<T> >
   */
   Plane3D(const Vector &n, const Point &p) { BuildFromVecAndPoint(n, p); }

   /**
    Construct from a generic DisplacementVector3D (normal vector) and PositionVector3D (point coplanar to
    the plane)
    \param n normal expressed as a generic ROOT::Math::DisplacementVector3D
    \param p point  expressed as a generic ROOT::Math::PositionVector3D
   */
   template <class T1, class T2, class U>
   Plane3D(const DisplacementVector3D<T1, U> &n, const PositionVector3D<T2, U> &p)
   {
      BuildFromVecAndPoint(Vector(n), Point(p));
   }

   /**
    constructor from three Cartesian point belonging to the plane
    \param p1 point1  expressed as a generic ROOT::Math::PositionVector3D
    \param p2 point2  expressed as a generic ROOT::Math::PositionVector3D
    \param p3 point3  expressed as a generic ROOT::Math::PositionVector3D
   */
   Plane3D(const Point &p1, const Point &p2, const Point &p3) { BuildFrom3Points(p1, p2, p3); }

   /**
    constructor from three generic point belonging to the plane
    \param p1 point1 expressed as  ROOT::Math::DisplacementVector3D<Cartesian3D<T> >
    \param p2 point2 expressed as  ROOT::Math::DisplacementVector3D<Cartesian3D<T> >
    \param p3 point3 expressed as  ROOT::Math::DisplacementVector3D<Cartesian3D<T> >
   */
   template <class T1, class T2, class T3, class U>
   Plane3D(const PositionVector3D<T1, U> &p1, const PositionVector3D<T2, U> &p2, const PositionVector3D<T3, U> &p3)
   {
      BuildFrom3Points(Point(p1.X(), p1.Y(), p1.Z()), Point(p2.X(), p2.Y(), p2.Z()), Point(p3.X(), p3.Y(), p3.Z()));
   }

   // compiler-generated copy ctor and dtor are fine.
   Plane3D(const Plane3D &) = default;

   // ------ assignment ------

   /**
      Assignment operator from other Plane3D class
   */
   Plane3D &operator=(const Plane3D &) = default;

   /**
      Return the a coefficient of the plane equation \f$ a*x + b*y + c*z + d = 0 \f$. It is also the
      x-component of the vector perpendicular to the plane.
   */
   Scalar A() const { return fA; }

   /**
      Return the b coefficient of the plane equation \f$ a*x + b*y + c*z + d = 0 \f$. It is also the
      y-component of the vector perpendicular to the plane
   */
   Scalar B() const { return fB; }

   /**
      Return the c coefficient of the plane equation \f$ a*x + b*y + c*z + d = 0 \f$. It is also the
      z-component of the vector perpendicular to the plane
   */
   Scalar C() const { return fC; }

   /**
      Return the d coefficient of the plane equation \f$ a*x + b*y + c*z + d = 0 \f$. It is also
      the distance from the origin (HesseDistance)
   */
   Scalar D() const { return fD; }

   /**
      Return normal vector to the plane as Cartesian DisplacementVector
   */
   Vector Normal() const { return Vector(fA, fB, fC); }

   /**
    Return the Hesse Distance (distance from the origin) of the plane or
    the d coefficient expressed in normalize form
   */
   Scalar HesseDistance() const { return fD; }

   /**
    Return the signed distance to a Point.
    The distance is signed positive if the Point is in the same side of the
    normal vector to the plane.
    \param p Point expressed in Cartesian Coordinates
    */
   Scalar Distance(const Point &p) const { return fA * p.X() + fB * p.Y() + fC * p.Z() + fD; }

   /**
    Return the distance to a Point described with generic coordinates
    \param p Point expressed as generic ROOT::Math::PositionVector3D
    */
   template <class T1, class U>
   Scalar Distance(const PositionVector3D<T1, U> &p) const
   {
      return Distance(Point(p.X(), p.Y(), p.Z()));
   }

   /**
    Return the projection of a Cartesian point to a plane
    \param p Point expressed as PositionVector3D<Cartesian3D<T> >
    */
   Point ProjectOntoPlane(const Point &p) const
   {
      const Scalar d = Distance(p);
      return XYZPoint(p.X() - fA * d, p.Y() - fB * d, p.Z() - fC * d);
   }

   /**
    Return the projection of a point to a plane
    \param p Point expressed as generic ROOT::Math::PositionVector3D
    */
   template <class T1, class U>
   PositionVector3D<T1, U> ProjectOntoPlane(const PositionVector3D<T1, U> &p) const
   {
      const Point pxyz = ProjectOntoPlane(Point(p.X(), p.Y(), p.Z()));
      return PositionVector3D<T, U>(pxyz.X(), pxyz.Y(), pxyz.Z());
   }

   // ------------------- Equality -----------------

   /**
      Exact equality
   */
   bool operator==(const Plane3D &rhs) const { return (fA == rhs.fA && fB == rhs.fB && fC == rhs.fC && fD == rhs.fD); }
   bool operator!=(const Plane3D &rhs) const { return !(operator==(rhs)); }

protected:
   /**
      Normalize the normal (a,b,c) plane components
   */
   template <typename SCALAR = T, typename std::enable_if<std::is_arithmetic<SCALAR>::value>::type * = nullptr>
   void Normalize()
   {
      // normalize the plane
      const SCALAR s = std::sqrt(fA * fA + fB * fB + fC * fC);
      // what to do if s = 0 ?
      if (s == SCALAR(0)) {
         fD = SCALAR(0);
      } else {
         const SCALAR w = Scalar(1) / s;
         fA *= w;
         fB *= w;
         fC *= w;
         fD *= w;
      }
   }

   /**
     Normalize the normal (a,b,c) plane components
   */
   template <typename SCALAR = T, typename std::enable_if<!std::is_arithmetic<SCALAR>::value>::type * = nullptr>
   void Normalize()
   {
      // normalize the plane
      SCALAR s = std::sqrt(fA * fA + fB * fB + fC * fC);
      // what to do if s = 0 ?
      const auto m = (s == SCALAR(0));
      // set zero entries to 1 in the vector to avoid /0 later on
      s(m)           = SCALAR(1);
      fD(m)          = SCALAR(0);
      const SCALAR w = SCALAR(1) / s;
      fA *= w;
      fB *= w;
      fC *= w;
      fD *= w;
   }

private:
   // internal method to construct class from a vector and a point
   void BuildFromVecAndPoint(const Vector &n, const Point &p)
   {
      // build from a normal vector and a point
      fA = n.X();
      fB = n.Y();
      fC = n.Z();
      fD = -n.Dot(p);
      Normalize();
   }

   // internal method to construct class from 3 points
   void BuildFrom3Points(const Point &p1, const Point &p2, const Point &p3)
   {
      // plane from thre points
      // normal is (x3-x1) cross (x2 -x1)
      const Vector n = (p2 - p1).Cross(p3 - p1);
      fA             = n.X();
      fB             = n.Y();
      fC             = n.Z();
      fD             = -n.Dot(p1);
      Normalize();
   }

   // plane data members the four scalar which  satisfies fA*x + fB*y + fC*z + fD = 0
   // for every point (x,y,z) belonging to the plane.
   // fA**2 + fB**2 + fC** =1 plane is stored in normalized form
   Scalar fA;
   Scalar fB;
   Scalar fC;
   Scalar fD;

   };  // Plane3D<>

   /**
      Stream Output and Input
   */
   // TODO - I/O should be put in the manipulator form
   template <typename T>
   std::ostream &operator<<(std::ostream &os, const Plane3D<T> &p)
   {
      os << "\n"
         << p.Normal().X() << "  " << p.Normal().Y() << "  " << p.Normal().Z() << "  " << p.HesseDistance() << "\n";
      return os;
   }

   } // end namespace Impl

   // typedefs for double and float versions
   typedef Impl::Plane3D<double> Plane3D;
   typedef Impl::Plane3D<float>  Plane3DF;

} // end namespace Math

} // end namespace ROOT


#endif
