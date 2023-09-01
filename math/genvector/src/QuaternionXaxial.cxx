// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Implementation file for quaternion times other non-axial rotations.
// Decoupled from main Quaternion implementations.
//
// Created by: Mark Fischler Tues July 19,  2005
//
// Last update: $Id$
//
#include "Math/GenVector/Quaternion.h"

namespace ROOT {

namespace Math {


// Although the same technique would work with axial rotations,
// we know that two of the four quaternion components will be zero,
// and we exploit that knowledge:

Quaternion Quaternion::operator * (const RotationX  & rx) const {
   // combination with a RotationX
   Quaternion q(rx);
   return Quaternion (
                      U()*q.U() - I()*q.I()
                      , I()*q.U() + U()*q.I()
                      , J()*q.U() + K()*q.I()
                      , K()*q.U() - J()*q.I()
                      );
}

Quaternion Quaternion::operator * (const RotationY  & ry) const {
   // combination with a RotationY
   Quaternion q(ry);
   return Quaternion (
                      U()*q.U() - J()*q.J()
                      , I()*q.U() - K()*q.J()
                      , J()*q.U() + U()*q.J()
                      , K()*q.U() + I()*q.J()
                      );
}

Quaternion Quaternion::operator * (const RotationZ  & rz) const {
   // combination with a RotationZ
   Quaternion q(rz);
   return Quaternion (
                      U()*q.U() - K()*q.K()
                      , I()*q.U() + J()*q.K()
                      , J()*q.U() - I()*q.K()
                      , K()*q.U() + U()*q.K()
                      );
}

Quaternion
operator * ( RotationX const & r, Quaternion const & q ) {
   return Quaternion(r) * q;  // TODO: improve performance
}

Quaternion
operator * ( RotationY const & r, Quaternion const & q ) {
   return Quaternion(r) * q;  // TODO: improve performance
}

Quaternion
operator * ( RotationZ const & r, Quaternion const & q ) {
   return Quaternion(r) * q;  // TODO: improve performance
}


} //namespace Math
} //namespace ROOT
