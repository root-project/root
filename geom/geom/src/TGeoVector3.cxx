// @(#)root/geom:$Id$
// Author: Andrei Gheata   15/01/2020

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class  TGeoVector3
\ingroup Geometry_classes
Simple 3-vector representation
*/

#include "TGeoVector3.h"

std::ostream &operator<<(std::ostream &os, ROOT::Geom::Vertex_t const &vec)
{
   os << "{" << vec[0] << ", " << vec[1] << ", " << vec[2] << "}";
   return os;
}
