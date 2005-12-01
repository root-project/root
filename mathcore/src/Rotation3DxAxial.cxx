// @(#)root/mathcore:$Name:  $:$Id: Rotation3DxAxial.cxx,v 1.1 2005/09/18 17:33:47 brun Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

#include "Math/GenVector/Rotation3D.h"

#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"

namespace ROOT {

  namespace Math {

Rotation3D
Rotation3D::
operator * (const RotationX  & rx) const {
  Scalar s = rx.SinAngle();
  Scalar c = rx.CosAngle();
  return Rotation3D
  (
    fM[XX],   fM[XY]*c + fM[XZ]*s,   fM[XZ]*c - fM[XY]*s 
  , fM[YX],   fM[YY]*c + fM[YZ]*s,   fM[YZ]*c - fM[YY]*s 
  , fM[ZX],   fM[ZY]*c + fM[ZZ]*s,   fM[ZZ]*c - fM[ZY]*s
  ); 
}

Rotation3D
Rotation3D::
operator * (const RotationY  & ry) const {
  Scalar s = ry.SinAngle();
  Scalar c = ry.CosAngle();
  return Rotation3D
  (
    fM[XX]*c - fM[XZ]*s,   fM[XY],   fM[XX]*s + fM[XZ]*c
  , fM[YX]*c - fM[YZ]*s,   fM[YY],   fM[YX]*s + fM[YZ]*c
  , fM[ZX]*c - fM[ZZ]*s,   fM[ZY],   fM[ZX]*s + fM[ZZ]*c
  ); 
}


Rotation3D
Rotation3D::
operator * (const RotationZ  & rz) const {
  Scalar s = rz.SinAngle();
  Scalar c = rz.CosAngle();
  return Rotation3D
  (
    fM[XX]*c + fM[XY]*s, fM[XY]*c - fM[XX]*s,   fM[XZ] 
  , fM[YX]*c + fM[YY]*s, fM[YY]*c - fM[YX]*s,	fM[YZ] 
  , fM[ZX]*c + fM[ZY]*s, fM[ZY]*c - fM[ZX]*s,	fM[ZZ] 
  ); 
}

Rotation3D operator* (RotationX const & r1, Rotation3D const & r2) {
  // TODO -- recode for much better efficiency!
  return Rotation3D(r1)*r2;
}

Rotation3D operator* (RotationY const & r1, Rotation3D const & r2) {
  // TODO -- recode for much better efficiency!
  return Rotation3D(r1)*r2;
}

Rotation3D operator* (RotationZ const & r1, Rotation3D const & r2) {
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
