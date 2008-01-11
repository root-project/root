// @(#)root/mathcore:$Id$
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

      double x1x = x1.x();      double x1y = x1.y();      double x1z = x1.z();
      double y1x = y1.x();      double y1y = y1.y();      double y1z = y1.z();
      double z1x = z1.x();      double z1y = z1.y();      double z1z = z1.z();

      double x2x = x2.x();      double x2y = x2.y();      double x2z = x2.z();
      double y2x = y2.x();      double y2y = y2.y();      double y2z = y2.z();
      double z2x = z2.x();      double z2y = z2.y();      double z2z = z2.z();
      
      double detxx =  (y1y *z1z  - z1y *y1z );
      double detxy = -(y1x *z1z  - z1x *y1z );
      double detxz =  (y1x *z1y  - z1x *y1y );
      double detyx = -(x1y *z1z  - z1y *x1z );
      double detyy =  (x1x *z1z  - z1x *x1z );
      double detyz = -(x1x *z1y  - z1x *x1y );
      double detzx =  (x1y *y1z  - y1y *x1z );
      double detzy = -(x1x *y1z  - y1x *x1z );
      double detzz =  (x1x *y1y  - y1x *x1y );
      
      double txx = x2x *detxx + y2x *detyx + z2x *detzx;
      double txy = x2x *detxy + y2x *detyy + z2x *detzy;
      double txz = x2x *detxz + y2x *detyz + z2x *detzz;
      double tyx = x2y *detxx + y2y *detyx + z2y *detzx;
      double tyy = x2y *detxy + y2y *detyy + z2y *detzy;
      double tyz = x2y *detxz + y2y *detyz + z2y *detzz;
      double tzx = x2z *detxx + y2z *detyx + z2z *detzx;
      double tzy = x2z *detxy + y2z *detyy + z2z *detzy;
      double tzz = x2z *detxz + y2z *detyz + z2z *detzz;
      
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
   //        and even the human readable form needs formatting improvements
   
   double m[12];
   t.GetComponents(m, m+12);
   os << "\n" << m[0] << "  " << m[1] << "  " << m[2] << "  " << m[3] ;
   os << "\n" << m[4] << "  " << m[5] << "  " << m[6] << "  " << m[7] ;
   os << "\n" << m[8] << "  " << m[9] << "  " << m[10]<< "  " << m[11] << "\n";
   return os;
}

}  // end namespace Math
}  // end namespace ROOT
