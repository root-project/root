// @(#)root/mathcore:$Id$
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
#include "Math/GenVector/RotationZYXfwd.h"


namespace ROOT {
namespace Math {


namespace gv_detail {

// flag a link time error when a wrong conversion is instantiated
   struct ERROR_This_Rotation_Conversion_is_NOT_Supported {
      ERROR_This_Rotation_Conversion_is_NOT_Supported();
   };
template<class R1, class R2>
void convert( R1 const &, R2 const) {
   ERROR_This_Rotation_Conversion_is_NOT_Supported();
}

// ----------------------------------------------------------------------
// conversions from Rotation3D
/**
   conversion functions from 3D rotation.
   For converting to Euler Angles see
   <A><HREF="http://www.cern.ch/mathlibs/documents/eulerAngleComputation.pdf">here</A>
 */

void convert( Rotation3D const & from, AxisAngle   & to);
void convert( Rotation3D const & from, EulerAngles & to);
void convert( Rotation3D const & from, Quaternion  & to);
void convert( Rotation3D const & from, RotationZYX & to);


// ----------------------------------------------------------------------
// conversions from AxisAngle

void convert( AxisAngle const & from, Rotation3D  & to);
void convert( AxisAngle const & from, EulerAngles & to);
void convert( AxisAngle const & from, Quaternion  & to);
void convert( AxisAngle const & from, RotationZYX & to);


// ----------------------------------------------------------------------
// conversions from EulerAngles

void convert( EulerAngles const & from, Rotation3D  & to);
void convert( EulerAngles const & from, AxisAngle   & to);
void convert( EulerAngles const & from, Quaternion  & to);
void convert( EulerAngles const & from, RotationZYX & to);


// ----------------------------------------------------------------------
// conversions from Quaternion

void convert( Quaternion const & from, Rotation3D  & to);
void convert( Quaternion const & from, AxisAngle   & to);
void convert( Quaternion const & from, EulerAngles & to);
void convert( Quaternion const & from, RotationZYX & to);

// ----------------------------------------------------------------------
// conversions from RotationZYX

void convert( RotationZYX const & from, Rotation3D  & to);
void convert( RotationZYX const & from, AxisAngle   & to);
void convert( RotationZYX const & from, EulerAngles & to);
void convert( RotationZYX const & from, Quaternion  & to);


// ----------------------------------------------------------------------
// conversions from RotationX

void convert( RotationX const & from, Rotation3D  & to);
void convert( RotationX const & from, RotationZYX & to);
void convert( RotationX const & from, AxisAngle   & to);
void convert( RotationX const & from, EulerAngles & to);
void convert( RotationX const & from, Quaternion  & to);


// ----------------------------------------------------------------------
// conversions from RotationY

void convert( RotationY const & from, Rotation3D  & to);
void convert( RotationY const & from, RotationZYX & to);
void convert( RotationY const & from, AxisAngle   & to);
void convert( RotationY const & from, EulerAngles & to);
void convert( RotationY const & from, Quaternion  & to);


// ----------------------------------------------------------------------
// conversions from RotationZ

void convert( RotationZ const & from, Rotation3D  & to);
void convert( RotationZ const & from, RotationZYX & to);
void convert( RotationZ const & from, AxisAngle   & to);
void convert( RotationZ const & from, EulerAngles & to);
void convert( RotationZ const & from, Quaternion  & to);


} //namespace gv_detail
} //namespace Math
} //namespace ROOT

#endif // ROOT_Math_GenVector_3DConversions
