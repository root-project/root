// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team and                     *
 *                      FNAL LCG ROOT MathLib Team                    *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header source file for CoordinateSystemTag's classes
//
// Created by: Lorenzo Moneta  at Wed Apr 05 2006
//
//

#ifndef ROOT_MathX_GenVectorX_CoordinateSystemTags
#define ROOT_MathX_GenVectorX_CoordinateSystemTags 1

#include "MathX/GenVectorX/AccHeaders.h"

namespace ROOT {

namespace ROOT_MATH_ARCH {

//__________________________________________________________________________________________
/**
   DefaultCoordinateSystemTag
   Default tag for identifying any coordinate system

   @ingroup GenVectorX

   @see GenVectorX
*/

class DefaultCoordinateSystemTag {};

//__________________________________________________________________________________________
/**
   Tag for identifying vectors based on a global coordinate system

   @ingroup GenVectorX

   @see GenVectorX
*/
class GlobalCoordinateSystemTag {};

//__________________________________________________________________________________________
/**
   Tag for identifying vectors based on a local coordinate system

   @ingroup GenVectorX

   @see GenVectorX
*/
class LocalCoordinateSystemTag {};

} // namespace ROOT_MATH_ARCH

} // namespace ROOT

#endif
