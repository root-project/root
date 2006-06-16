// @(#)root/mathcore:$Name:  $:$Id: Transform3D.cxx,v 1.9 2006/06/15 16:23:44 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// implementation file for class Transform3D
//
// Created by: Lorenzo Moneta  October 27 2005
//
//
#include "Math/GenVector/GenVectorIO.h"

#include "Math/GenVector/Transform3D.h"
#include "Math/GenVector/Plane3D.h"

#include <cmath>
#include <algorithm>




namespace ROOT {

   namespace Math {


    typedef Transform3D::Point  XYZPoint; 
    typedef Transform3D::Vector XYZVector; 


// ========== Constructors and Assignment =====================



// construct from two ref frames
Transform3D::Transform3D(const XYZPoint & fr0, const XYZPoint & fr1, const XYZPoint & fr2,
                         const XYZPoint & to0, const XYZPoint & to1, const XYZPoint & to2 )
{
   // takes impl. from CLHEP ( E.Chernyaev). To be checked

   XYZVector x1,y1,z1, x2,y2,z2;
   x1 = (fr1 - fr0).Unit();
   y1 = (fr2 - fr0).Unit();
   x2 = (to1 - to0).Unit();
   y2 = (to2 - to0).Unit();

   //   C H E C K   A N G L E S

   double cos1, cos2;
   cos1 = x1.Dot(y1);
   cos2 = x2.Dot(y2);

   if (std::fabs(1.0-cos1) <= 0.000001 || std::fabs(1.0-cos2) <= 0.000001) {
      std::cerr << "Transform3D: Error : zero angle between axes" << std::endl;
      SetIdentity();
   } else {
      if (std::fabs(cos1-cos2) > 0.000001) {
         std::cerr << "Transform3D: Warning: angles between axes are not equal"
                   << std::endl;
       }

      //   F I N D   R O T A T I O N   M A T R I X

      z1 = (x1.Cross(y1)).Unit();
      y1  = z1.Cross(x1);

      z2 = (x2.Cross(y2)).Unit();
      y2  = z2.Cross(x2);

      double detxx =  (y1.y()*z1.z() - z1.y()*y1.z());
      double detxy = -(y1.x()*z1.z() - z1.x()*y1.z());
      double detxz =  (y1.x()*z1.y() - z1.x()*y1.y());
      double detyx = -(x1.y()*z1.z() - z1.y()*x1.z());
      double detyy =  (x1.x()*z1.z() - z1.x()*x1.z());
      double detyz = -(x1.x()*z1.y() - z1.x()*x1.y());
      double detzx =  (x1.y()*y1.z() - y1.y()*x1.z());
      double detzy = -(x1.x()*y1.z() - y1.x()*x1.z());
      double detzz =  (x1.x()*y1.y() - y1.x()*x1.y());

      double txx = x2.x()*detxx + y2.x()*detyx + z2.x()*detzx;
      double txy = x2.x()*detxy + y2.x()*detyy + z2.x()*detzy;
      double txz = x2.x()*detxz + y2.x()*detyz + z2.x()*detzz;
      double tyx = x2.y()*detxx + y2.y()*detyx + z2.y()*detzx;
      double tyy = x2.y()*detxy + y2.y()*detyy + z2.y()*detzy;
      double tyz = x2.y()*detxz + y2.y()*detyz + z2.y()*detzz;
      double tzx = x2.z()*detxx + y2.z()*detyx + z2.z()*detzx;
      double tzy = x2.z()*detxy + y2.z()*detyy + z2.z()*detzy;
      double tzz = x2.z()*detxz + y2.z()*detyz + z2.z()*detzz;

      //   S E T    T R A N S F O R M A T I O N

      double dx1 = fr0.x(), dy1 = fr0.y(), dz1 = fr0.z();
      double dx2 = to0.x(), dy2 = to0.y(), dz2 = to0.z();

      SetComponents(txx, txy, txz, dx2-txx*dx1-txy*dy1-txz*dz1,
                    tyx, tyy, tyz, dy2-tyx*dx1-tyy*dy1-tyz*dz1,
                    tzx, tzy, tzz, dz2-tzx*dx1-tzy*dy1-tzz*dz1);
   }
}


// inversion (from CLHEP)
void Transform3D::Invert()
{
   //
   // Name: Transform3D::inverse                     Date:    24.09.96
   // Author: E.Chernyaev (IHEP/Protvino)            Revised:
   //
   // Function: Find inverse affine transformation.

   double detxx = fM[kYY]*fM[kZZ] - fM[kYZ]*fM[kZY];
   double detxy = fM[kYX]*fM[kZZ] - fM[kYZ]*fM[kZX];
   double detxz = fM[kYX]*fM[kZY] - fM[kYY]*fM[kZX];
   double det   = fM[kXX]*detxx - fM[kXY]*detxy + fM[kXZ]*detxz;
   if (det == 0) {
      std::cerr << "Transform3D::inverse error: zero determinant" << std::endl;
      return;
   }
   det = 1./det; detxx *= det; detxy *= det; detxz *= det;
   double detyx = (fM[kXY]*fM[kZZ] - fM[kXZ]*fM[kZY] )*det;
   double detyy = (fM[kXX]*fM[kZZ] - fM[kXZ]*fM[kZX] )*det;
   double detyz = (fM[kXX]*fM[kZY] - fM[kXY]*fM[kZX] )*det;
   double detzx = (fM[kXY]*fM[kYZ] - fM[kXZ]*fM[kYY] )*det;
   double detzy = (fM[kXX]*fM[kYZ] - fM[kXZ]*fM[kYX] )*det;
   double detzz = (fM[kXX]*fM[kYY] - fM[kXY]*fM[kYX] )*det;
   SetComponents
      (detxx, -detyx,  detzx, -detxx*fM[kDX]+detyx*fM[kDY]-detzx*fM[kDZ],
      -detxy,  detyy, -detzy,  detxy*fM[kDX]-detyy*fM[kDY]+detzy*fM[kDZ],
       detxz, -detyz,  detzz, -detxz*fM[kDX]+detyz*fM[kDY]-detzz*fM[kDZ]);
}


// get rotations and translations
void Transform3D::GetDecomposition ( Rotation3D &r, XYZVector &v) const
{
  // decompose a trasfomation in a 3D rotation and in a 3D vector (cartesian coordinates) 
   r.SetComponents( fM[kXX], fM[kXY], fM[kXZ],
                    fM[kYX], fM[kYY], fM[kYZ],
                    fM[kZX], fM[kZY], fM[kZZ] );

   v.SetCoordinates( fM[kDX], fM[kDY], fM[kDZ] );
}

// transformation on Position Vector (rotation + translations)
XYZPoint Transform3D::operator() (const XYZPoint & p) const
{
   // pass through rotation class (could be implemented directly to be faster)

   Rotation3D r;
   XYZVector  t;
   GetDecomposition(r, t);
   XYZPoint pnew = r(p);
   pnew += t;
   return pnew;
}

// transformation on Displacement Vector (only rotation)
XYZVector Transform3D::operator() (const XYZVector & v) const
{
   // pass through rotation class ( could be implemented directly to be faster)

   Rotation3D r;
   XYZVector  t;
   GetDecomposition(r, t);
   // only rotation
   return r(v);
}

Transform3D & Transform3D::operator *= (const Transform3D  & t)
{
// combination of transformations

   SetComponents(fM[kXX]*t.fM[kXX]+fM[kXY]*t.fM[kYX]+fM[kXZ]*t.fM[kZX],
                 fM[kXX]*t.fM[kXY]+fM[kXY]*t.fM[kYY]+fM[kXZ]*t.fM[kZY],
                 fM[kXX]*t.fM[kXZ]+fM[kXY]*t.fM[kYZ]+fM[kXZ]*t.fM[kZZ],
                 fM[kXX]*t.fM[kDX]+fM[kXY]*t.fM[kDY]+fM[kXZ]*t.fM[kDZ]+fM[kDX],

                 fM[kYX]*t.fM[kXX]+fM[kYY]*t.fM[kYX]+fM[kYZ]*t.fM[kZX],
                 fM[kYX]*t.fM[kXY]+fM[kYY]*t.fM[kYY]+fM[kYZ]*t.fM[kZY],
                 fM[kYX]*t.fM[kXZ]+fM[kYY]*t.fM[kYZ]+fM[kYZ]*t.fM[kZZ],
                 fM[kYX]*t.fM[kDX]+fM[kYY]*t.fM[kDY]+fM[kYZ]*t.fM[kDZ]+fM[kDY],

                 fM[kZX]*t.fM[kXX]+fM[kZY]*t.fM[kYX]+fM[kZZ]*t.fM[kZX],
                 fM[kZX]*t.fM[kXY]+fM[kZY]*t.fM[kYY]+fM[kZZ]*t.fM[kZY],
                 fM[kZX]*t.fM[kXZ]+fM[kZY]*t.fM[kYZ]+fM[kZZ]*t.fM[kZZ],
                 fM[kZX]*t.fM[kDX]+fM[kZY]*t.fM[kDY]+fM[kZZ]*t.fM[kDZ]+fM[kDZ]);

   return *this;
}

void Transform3D::SetIdentity()
{
   //set identity ( identity rotation and zero translation)
   fM[kXX] = 1.0;  fM[kXY] = 0.0; fM[kXZ] = 0.0; fM[kDX] = 0.0;
   fM[kYX] = 0.0;  fM[kYY] = 1.0; fM[kYZ] = 0.0; fM[kDY] = 0.0;
   fM[kZX] = 0.0;  fM[kZY] = 0.0; fM[kZZ] = 1.0; fM[kDZ] = 0.0;
}


void Transform3D::AssignFrom (const Rotation3D  & r,  const XYZVector & v)
{
  // assignment  from rotation + translation

   double rotData[9];
   r.GetComponents(rotData, rotData +9);
   // first raw
   for (int i = 0; i < 3; ++i)
      fM[i] = rotData[i];
   // second raw
   for (int i = 0; i < 3; ++i)
      fM[kYX+i] = rotData[3+i];
   // third raw
   for (int i = 0; i < 3; ++i)
      fM[kZX+i] = rotData[6+i];

   // translation data
   double vecData[3];
   v.GetCoordinates(vecData, vecData+3);
   fM[kDX] = vecData[0];
   fM[kDY] = vecData[1];
   fM[kDZ] = vecData[2];
 }


void Transform3D::AssignFrom(const Rotation3D & r)
{
  // assign from only a rotation  (null translation)
   double rotData[9];
   r.GetComponents(rotData, rotData +9);
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j)
        fM[4*i + j] = rotData[3*i+j];
      // empty vector data
      fM[4*i + 3] = 0;
   }
}

void Transform3D::AssignFrom(const XYZVector & v)
{
  // assign from a translation only (identity rotations)
   fM[kXX] = 1.0;  fM[kXY] = 0.0; fM[kXZ] = 0.0; fM[kDX] = v.X();
   fM[kYX] = 0.0;  fM[kYY] = 1.0; fM[kYZ] = 0.0; fM[kDY] = v.Y();
   fM[kZX] = 0.0;  fM[kZY] = 0.0; fM[kZZ] = 1.0; fM[kDZ] = v.Z();
}

Plane3D Transform3D::operator() (const Plane3D & plane) const
{
  // transformations on a 3D plane
   XYZVector n = plane.Normal();
   // take a point on the plane. Use origin projection on the plane
   // ( -ad, -bd, -cd) if (a**2 + b**2 + c**2 ) = 1
   double d = plane.HesseDistance();
   XYZPoint p( - d * n.X() , - d *n.Y(), -d *n.Z() );
   return Plane3D ( operator() (n), operator() (p) );
}

std::ostream & operator<< (std::ostream & os, const Transform3D & t)
{
   // TODO - this will need changing for machine-readable issues
   //        and even the human readable form needs formatiing improvements

   double m[12];
   t.GetComponents(m, m+12);
   os << "\n" << m[0] << "  " << m[1] << "  " << m[2] << "  " << m[3] ;
   os << "\n" << m[4] << "  " << m[5] << "  " << m[6] << "  " << m[7] ;
   os << "\n" << m[8] << "  " << m[9] << "  " << m[10]<< "  " << m[11] << "\n";
   return os;
}

}  // end namespace Math
}  // end namespace ROOT
