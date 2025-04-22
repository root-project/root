// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file Point2Dfwd
//
// Created by: Lorenzo Moneta  at Mon Apr 16 2007
//

#ifndef ROOT_MathX_Point2Dfwd
#define ROOT_MathX_Point2Dfwd 1

#include "MathX/GenVectorX/AccHeaders.h"

#include "MathX/GenVectorX/MathHeaders.h"

// forward declareations of position vectors (Points) and type defs definitions

namespace ROOT {

namespace ROOT_MATH_ARCH {

template <class CoordSystem, class Tag>
class PositionVector2D;

template <typename T>
class Cartesian2D;
template <typename T>
class Polar2D;

class DefaultCoordinateSystemTag;

/**
   2D Point based on the cartesian coordinates x,y,z in double precision
*/
typedef PositionVector2D<Cartesian2D<double>, DefaultCoordinateSystemTag> XYPoint;
typedef XYPoint XYPointD;

/**
   2D Point based on the cartesian coordinates x,y,z in single precision
*/
typedef PositionVector2D<Cartesian2D<float>, DefaultCoordinateSystemTag> XYPointF;

/**
   2D Point based on the polar coordinates rho, theta, phi in double precision.
*/
typedef PositionVector2D<Polar2D<double>, DefaultCoordinateSystemTag> Polar2DPoint;
typedef Polar2DPoint Polar2DPointD;

/**
   2D Point based on the polar coordinates rho, theta, phi in single precision.
*/
typedef PositionVector2D<Polar2D<float>, DefaultCoordinateSystemTag> Polar2DPointF;

} // namespace ROOT_MATH_ARCH

} // end namespace ROOT

#endif /* ROOT_MathX_Point2Dfwd  */
