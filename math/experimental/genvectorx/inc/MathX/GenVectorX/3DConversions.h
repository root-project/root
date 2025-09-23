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
#ifndef ROOT_MathX_GenVectorX_3DConversions
#define ROOT_MathX_GenVectorX_3DConversions 1

#include "MathX/GenVectorX/Rotation3Dfwd.h"
#include "MathX/GenVectorX/AxisAnglefwd.h"
#include "MathX/GenVectorX/EulerAnglesfwd.h"
#include "MathX/GenVectorX/Quaternionfwd.h"
#include "MathX/GenVectorX/RotationXfwd.h"
#include "MathX/GenVectorX/RotationYfwd.h"
#include "MathX/GenVectorX/RotationZfwd.h"
#include "MathX/GenVectorX/RotationZYXfwd.h"

#include "MathX/GenVectorX/AccHeaders.h"

#include "MathX/GenVectorX/MathHeaders.h"

namespace ROOT {
namespace ROOT_MATH_ARCH {

namespace gv_detail {

// flag a link time error when a wrong conversion is instantiated
struct ERROR_This_Rotation_Conversion_is_NOT_Supported {
   ERROR_This_Rotation_Conversion_is_NOT_Supported();
};
template <class R1, class R2>
void convert(R1 const &, R2 const)
{
   ERROR_This_Rotation_Conversion_is_NOT_Supported();
}

// ----------------------------------------------------------------------
// conversions from Rotation3D
/**
   conversion functions from 3D rotation.
 */

void convert(Rotation3D const &from, AxisAngle &to);
void convert(Rotation3D const &from, EulerAngles &to);
void convert(Rotation3D const &from, Quaternion &to);
void convert(Rotation3D const &from, RotationZYX &to);

// ----------------------------------------------------------------------
// conversions from AxisAngle

void convert(::ROOT::ROOT_MATH_ARCH::AxisAngle const &from, ::ROOT::ROOT_MATH_ARCH::Rotation3D &to);
void convert(::ROOT::ROOT_MATH_ARCH::AxisAngle const &from, ::ROOT::ROOT_MATH_ARCH::EulerAngles &to);
void convert(::ROOT::ROOT_MATH_ARCH::AxisAngle const &from, ::ROOT::ROOT_MATH_ARCH::Quaternion &to);
void convert(::ROOT::ROOT_MATH_ARCH::AxisAngle const &from, ::ROOT::ROOT_MATH_ARCH::RotationZYX &to);

// ----------------------------------------------------------------------
// conversions from EulerAngles

void convert(::ROOT::ROOT_MATH_ARCH::EulerAngles const &from, ::ROOT::ROOT_MATH_ARCH::Rotation3D &to);
void convert(::ROOT::ROOT_MATH_ARCH::EulerAngles const &from, ::ROOT::ROOT_MATH_ARCH::AxisAngle &to);
void convert(::ROOT::ROOT_MATH_ARCH::EulerAngles const &from, ::ROOT::ROOT_MATH_ARCH::Quaternion &to);
void convert(::ROOT::ROOT_MATH_ARCH::EulerAngles const &from, ::ROOT::ROOT_MATH_ARCH::RotationZYX &to);

// ----------------------------------------------------------------------
// conversions from Quaternion

void convert(::ROOT::ROOT_MATH_ARCH::Quaternion const &from, ::ROOT::ROOT_MATH_ARCH::Rotation3D &to);
void convert(::ROOT::ROOT_MATH_ARCH::Quaternion const &from, ::ROOT::ROOT_MATH_ARCH::AxisAngle &to);
void convert(::ROOT::ROOT_MATH_ARCH::Quaternion const &from, ::ROOT::ROOT_MATH_ARCH::EulerAngles &to);
void convert(::ROOT::ROOT_MATH_ARCH::Quaternion const &from, ::ROOT::ROOT_MATH_ARCH::RotationZYX &to);

// ----------------------------------------------------------------------
// conversions from RotationZYX

void convert(::ROOT::ROOT_MATH_ARCH::RotationZYX const &from, ::ROOT::ROOT_MATH_ARCH::Rotation3D &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationZYX const &from, ::ROOT::ROOT_MATH_ARCH::AxisAngle &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationZYX const &from, ::ROOT::ROOT_MATH_ARCH::EulerAngles &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationZYX const &from, ::ROOT::ROOT_MATH_ARCH::Quaternion &to);

// ----------------------------------------------------------------------
// conversions from RotationX

void convert(::ROOT::ROOT_MATH_ARCH::RotationX const &from, ::ROOT::ROOT_MATH_ARCH::Rotation3D &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationX const &from, ::ROOT::ROOT_MATH_ARCH::RotationZYX &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationX const &from, ::ROOT::ROOT_MATH_ARCH::AxisAngle &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationX const &from, ::ROOT::ROOT_MATH_ARCH::EulerAngles &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationX const &from, ::ROOT::ROOT_MATH_ARCH::Quaternion &to);

// ----------------------------------------------------------------------
// conversions from RotationY

void convert(::ROOT::ROOT_MATH_ARCH::RotationY const &from, ::ROOT::ROOT_MATH_ARCH::Rotation3D &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationY const &from, ::ROOT::ROOT_MATH_ARCH::RotationZYX &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationY const &from, ::ROOT::ROOT_MATH_ARCH::AxisAngle &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationY const &from, ::ROOT::ROOT_MATH_ARCH::EulerAngles &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationY const &from, ::ROOT::ROOT_MATH_ARCH::Quaternion &to);

// ----------------------------------------------------------------------
// conversions from RotationZ

void convert(::ROOT::ROOT_MATH_ARCH::RotationZ const &from, ::ROOT::ROOT_MATH_ARCH::Rotation3D &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationZ const &from, ::ROOT::ROOT_MATH_ARCH::RotationZYX &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationZ const &from, ::ROOT::ROOT_MATH_ARCH::AxisAngle &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationZ const &from, ::ROOT::ROOT_MATH_ARCH::EulerAngles &to);
void convert(::ROOT::ROOT_MATH_ARCH::RotationZ const &from, ::ROOT::ROOT_MATH_ARCH::Quaternion &to);

} // namespace gv_detail
} // namespace ROOT_MATH_ARCH
} // namespace ROOT

#endif // ROOT_MathX_GenVectorX_3DConversions
