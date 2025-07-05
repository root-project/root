// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005, LCG ROOT FNAL MathLib Team                    *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Source file for something else
//
// Created by: Mark Fischler Thurs July 7, 2005
//
// Last update: Wed Thurs July 7, 2005
//

#include "MathX/GenVectorX/3DDistances.h"

#include "MathX/GenVectorX/Rotation3D.h"
#include "MathX/GenVectorX/AxisAngle.h"
#include "MathX/GenVectorX/EulerAngles.h"
#include "MathX/GenVectorX/Quaternion.h"
#include "MathX/GenVectorX/RotationZYX.h"
#include "MathX/GenVectorX/RotationX.h"
#include "MathX/GenVectorX/RotationY.h"
#include "MathX/GenVectorX/RotationZ.h"

#include <cmath>

#include "MathX/GenVectorX/AccHeaders.h"

#include "MathX/GenVectorX/MathHeaders.h"

namespace ROOT {
namespace ROOT_MATH_ARCH {
namespace gv_detail {

enum ERotation3DMatrixIndex {
   kXX = 0,
   kXY = 1,
   kXZ = 2,
   kYX = 3,
   kYY = 4,
   kYZ = 5,
   kZX = 6,
   kZY = 7,
   kZZ = 8
};

// ----------------------------------------------------------------------
// distance from Rotation3D

double dist(Rotation3D const &from, Rotation3D const &to)
{ /*TODO better */
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(Rotation3D const &from, AxisAngle const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(Rotation3D const &from, EulerAngles const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(Rotation3D const &from, Quaternion const &to)
{
   return Quaternion(from).Distance(to);
}

double dist(Rotation3D const &from, RotationZYX const &to)
{ /*TODO better */
   return Quaternion(from).Distance(Quaternion(to));
}
double dist(Rotation3D const &from, RotationX const &to)
{ /*TODO better */
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(Rotation3D const &from, RotationY const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(Rotation3D const &from, RotationZ const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

// ----------------------------------------------------------------------
// distance from AxisAngle

double dist(AxisAngle const &from, Rotation3D const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(AxisAngle const &from, AxisAngle const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(AxisAngle const &from, EulerAngles const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(AxisAngle const &from, Quaternion const &to)
{
   return Quaternion(from).Distance(to);
}

double dist(AxisAngle const &from, RotationZYX const &to)
{ /*TODO better */
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(AxisAngle const &from, RotationX const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(AxisAngle const &from, RotationY const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(AxisAngle const &from, RotationZ const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

// ----------------------------------------------------------------------
// distance from EulerAngles

double dist(EulerAngles const &from, Rotation3D const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(EulerAngles const &from, AxisAngle const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(EulerAngles const &from, EulerAngles const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(EulerAngles const &from, Quaternion const &to)
{
   return Quaternion(from).Distance(to);
}

double dist(EulerAngles const &from, RotationZYX const &to)
{ /*TODO better */
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(EulerAngles const &from, RotationX const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(EulerAngles const &from, RotationY const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(EulerAngles const &from, RotationZ const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

// ----------------------------------------------------------------------
// distance from Quaternion

double dist(Quaternion const &from, Rotation3D const &to)
{
   return from.Distance(Quaternion(to));
}

double dist(Quaternion const &from, AxisAngle const &to)
{
   return from.Distance(Quaternion(to));
}

double dist(Quaternion const &from, EulerAngles const &to)
{
   return from.Distance(Quaternion(to));
}

double dist(Quaternion const &from, Quaternion const &to)
{
   return from.Distance(to);
}

double dist(Quaternion const &from, RotationZYX const &to)
{
   return from.Distance(Quaternion(to));
}

double dist(Quaternion const &from, RotationX const &to)
{ /*TODO*/
   return from.Distance(Quaternion(to));
}

double dist(Quaternion const &from, RotationY const &to)
{ /*TODO*/
   return from.Distance(Quaternion(to));
}

double dist(Quaternion const &from, RotationZ const &to)
{ /*TODO*/
   return from.Distance(Quaternion(to));
}

// ----------------------------------------------------------------------
// distance from RotationZYX

double dist(RotationZYX const &from, Rotation3D const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZYX const &from, AxisAngle const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZYX const &from, EulerAngles const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZYX const &from, Quaternion const &to)
{
   return Quaternion(from).Distance(to);
}

double dist(RotationZYX const &from, RotationZYX const &to)
{ /*TODO better */
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZYX const &from, RotationX const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZYX const &from, RotationY const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZYX const &from, RotationZ const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

// ----------------------------------------------------------------------
// distance from RotationX

double dist(RotationX const &from, Rotation3D const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationX const &from, AxisAngle const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationX const &from, EulerAngles const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationX const &from, Quaternion const &to)
{
   return Quaternion(from).Distance(to);
}

double dist(RotationX const &from, RotationZYX const &to)
{ /*TODO better */
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationX const &from, RotationX const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationX const &from, RotationY const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationX const &from, RotationZ const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

// ----------------------------------------------------------------------
// distance from RotationY

double dist(RotationY const &from, Rotation3D const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationY const &from, AxisAngle const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationY const &from, EulerAngles const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationY const &from, Quaternion const &to)
{
   return Quaternion(from).Distance(to);
}

double dist(RotationY const &from, RotationZYX const &to)
{ /*TODO better */
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationY const &from, RotationX const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationY const &from, RotationY const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationY const &from, RotationZ const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

// ----------------------------------------------------------------------
// distance from RotationZ

double dist(RotationZ const &from, Rotation3D const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZ const &from, AxisAngle const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZ const &from, EulerAngles const &to)
{
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZ const &from, Quaternion const &to)
{
   return Quaternion(from).Distance(to);
}

double dist(RotationZ const &from, RotationZYX const &to)
{ /*TODO better */
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZ const &from, RotationX const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZ const &from, RotationY const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

double dist(RotationZ const &from, RotationZ const &to)
{ /*TODO*/
   return Quaternion(from).Distance(Quaternion(to));
}

} // namespace gv_detail
} // namespace ROOT_MATH_ARCH
} // namespace ROOT
