// @(#)root/mathcore:$Name:  $:$Id: Rotation3DxAxial.cxxv 1.0 2005/06/23 12:00:00 moneta Exp $
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


} //namespace Math
} //namespace ROOT
