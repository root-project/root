// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

#include "Math/GenVector/Rotation3D.h"

#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"

namespace ROOT {

namespace Math {

Rotation3D Rotation3D::operator * (const RotationX  & rx) const {
   // combination of a Rotation3D with a RotationX
   Scalar s = rx.SinAngle();
   Scalar c = rx.CosAngle();
   return Rotation3D
      (
       fM[kXX],   fM[kXY]*c + fM[kXZ]*s,   fM[kXZ]*c - fM[kXY]*s 
       , fM[kYX],   fM[kYY]*c + fM[kYZ]*s,   fM[kYZ]*c - fM[kYY]*s 
       , fM[kZX],   fM[kZY]*c + fM[kZZ]*s,   fM[kZZ]*c - fM[kZY]*s
       ); 
}

Rotation3D Rotation3D::operator * (const RotationY  & ry) const {
   // combination of a Rotation3D with a RotationY
   Scalar s = ry.SinAngle();
   Scalar c = ry.CosAngle();
   return Rotation3D
      (
       fM[kXX]*c - fM[kXZ]*s,   fM[kXY],   fM[kXX]*s + fM[kXZ]*c
       , fM[kYX]*c - fM[kYZ]*s,   fM[kYY],   fM[kYX]*s + fM[kYZ]*c
       , fM[kZX]*c - fM[kZZ]*s,   fM[kZY],   fM[kZX]*s + fM[kZZ]*c
       ); 
}


Rotation3D Rotation3D::operator * (const RotationZ  & rz) const {
   // combination of a Rotation3D with a RotationZ
   Scalar s = rz.SinAngle();
   Scalar c = rz.CosAngle();
   return Rotation3D
      (
       fM[kXX]*c + fM[kXY]*s, fM[kXY]*c - fM[kXX]*s,   fM[kXZ] 
       , fM[kYX]*c + fM[kYY]*s, fM[kYY]*c - fM[kYX]*s,	fM[kYZ] 
       , fM[kZX]*c + fM[kZY]*s, fM[kZY]*c - fM[kZX]*s,	fM[kZZ] 
       ); 
}

Rotation3D operator* (RotationX const & r1, Rotation3D const & r2) {
   // combination of a RotationX with a Rotation3D 
   // TODO -- recode for much better efficiency!
   return Rotation3D(r1)*r2;
}

Rotation3D operator* (RotationY const & r1, Rotation3D const & r2) {
   // combination of a RotationY with a Rotation3D 
   // TODO -- recode for much better efficiency!
   return Rotation3D(r1)*r2;
}

Rotation3D operator* (RotationZ const & r1, Rotation3D const & r2) {
   // combination of a RotationZ with a Rotation3D 
   // TODO -- recode for much better efficiency!
   return Rotation3D(r1)*r2;
}

typedef Rotation3D::Scalar Scalar;

// Rx * Ry
Rotation3D operator* (RotationX const & rx, RotationY const & ry) {
   Scalar sx = rx.SinAngle();
   Scalar cx = rx.CosAngle();
   Scalar sy = ry.SinAngle();
   Scalar cy = ry.CosAngle();
   return Rotation3D
      (  cy     ,  0   ,    sy   , 
         sx*sy  , cx   , -sx*cy  ,
         -sy*cx  , sx   ,  cx*cy  ); 
}

// Rx * Rz
Rotation3D operator* (RotationX const & rx, RotationZ const & rz) {
   Scalar sx = rx.SinAngle();
   Scalar cx = rx.CosAngle();
   Scalar sz = rz.SinAngle();
   Scalar cz = rz.CosAngle();
   return Rotation3D
      (  cz     ,   -sz ,     0  , 
         cx*sz  , cx*cz ,   -sx  ,
         sx*sz  , cz*sx ,    cx  ); 
}

// Ry * Rx
Rotation3D operator* (RotationY const & ry, RotationX const & rx) {
   Scalar sx = rx.SinAngle();
   Scalar cx = rx.CosAngle();
   Scalar sy = ry.SinAngle();
   Scalar cy = ry.CosAngle();
   return Rotation3D
      (  cy     , sx*sy ,  sy*cx  , 
         0     ,    cx ,    -sx  ,
         -sy     , cy*sx ,  cx*cy  ); 
}

// Ry * Rz
Rotation3D operator* (RotationY const & ry, RotationZ const & rz) {
   Scalar sy = ry.SinAngle();
   Scalar cy = ry.CosAngle();
   Scalar sz = rz.SinAngle();
   Scalar cz = rz.CosAngle();
   return Rotation3D
      (  cy*cz  ,-cy*sz ,    sy  , 
         sz  ,    cz ,     0  ,
         -cz*sy  , sy*sz ,    cy  ); 
}

// Rz * Rx
Rotation3D operator* (RotationZ const & rz, RotationX const & rx) {
   Scalar sx = rx.SinAngle();
   Scalar cx = rx.CosAngle();
   Scalar sz = rz.SinAngle();
   Scalar cz = rz.CosAngle();
   return Rotation3D
      (     cz  ,-cx*sz , sx*sz  , 
            sz  , cx*cz ,-cz*sx  ,
            0  ,    sx ,    cx  ); 
}

// Rz * Ry
Rotation3D operator* (RotationZ const & rz, RotationY const & ry) {
   Scalar sy = ry.SinAngle();
   Scalar cy = ry.CosAngle();
   Scalar sz = rz.SinAngle();
   Scalar cz = rz.CosAngle();
   return Rotation3D
      (  cy*cz  ,   -sz , cz*sy  , 
         cy*sz  ,    cz , sy*sz  ,
         -sy  ,     0 ,    cy  ); 
}




} //namespace Math
} //namespace ROOT
