// @(#)root/mathcore:$Name:  $:$Id: 3DConversions.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005, LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Rotation in 3 dimensions, represented by 3x3 matrix
//
// Created by: Mark Fischler and Walter Brown Thurs July 7, 2005
//
// Last update: Wed Thurs July 7, 2005
//
#ifndef ROOT_Math_GenVector_3DConversions 
#define ROOT_Math_GenVector_3DConversions  1

#include "Math/GenVector/Rotation3Dfwd.h"
#include "Math/GenVector/AxisAnglefwd.h"
#include "Math/GenVector/EulerAnglesfwd.h"
#include "Math/GenVector/Quaternionfwd.h"
#include "Math/GenVector/RotationXfwd.h"
#include "Math/GenVector/RotationYfwd.h"
#include "Math/GenVector/RotationZfwd.h"


namespace ROOT {
namespace Math {
namespace gv_detail {


// ----------------------------------------------------------------------
// conversions from Rotation3D

void convert( Rotation3D const & from, AxisAngle   & to);
void convert( Rotation3D const & from, EulerAngles & to);
void convert( Rotation3D const & from, Quaternion  & to);


// ----------------------------------------------------------------------
// conversions from AxisAngle

void convert( AxisAngle const & from, Rotation3D  & to);
void convert( AxisAngle const & from, EulerAngles & to);
void convert( AxisAngle const & from, Quaternion  & to);


// ----------------------------------------------------------------------
// conversions from EulerAngles

void convert( EulerAngles const & from, Rotation3D  & to);
void convert( EulerAngles const & from, AxisAngle   & to);
void convert( EulerAngles const & from, Quaternion  & to);


// ----------------------------------------------------------------------
// conversions from Quaternion

void convert( Quaternion const & from, Rotation3D  & to);
void convert( Quaternion const & from, AxisAngle   & to);
void convert( Quaternion const & from, EulerAngles & to);


// ----------------------------------------------------------------------
// conversions from RotationX

void convert( RotationX const & from, Rotation3D  & to);
void convert( RotationX const & from, AxisAngle   & to);
void convert( RotationX const & from, EulerAngles & to);
void convert( RotationX const & from, Quaternion  & to);


// ----------------------------------------------------------------------
// conversions from RotationY

void convert( RotationY const & from, Rotation3D  & to);
void convert( RotationY const & from, AxisAngle   & to);
void convert( RotationY const & from, EulerAngles & to);
void convert( RotationY const & from, Quaternion  & to);


// ----------------------------------------------------------------------
// conversions from RotationZ

void convert( RotationZ const & from, Rotation3D  & to);
void convert( RotationZ const & from, AxisAngle   & to);
void convert( RotationZ const & from, EulerAngles & to);
void convert( RotationZ const & from, Quaternion  & to);


} //namespace gv_detail
} //namespace Math
} //namespace ROOT

#endif // ROOT_Math_GenVector_3DConversions 
