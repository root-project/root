/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ACTIVE_GEOMETRY_TRANSFORM_H
#define ACTIVE_GEOMETRY_TRANSFORM_H

namespace ROOT::Internal {
/// If a TGeometry has been instantiated, this function transforms points from a local coordinate
/// system to master. To decouple TGeometry from graf3d, this function pointer resides in graf3d,
/// but is written to from TGeometry.
[[maybe_unused]] inline void (*currentTGeometryTransformer)(double *, double *) = nullptr;
} // namespace ROOT::Internal

#endif