// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TPartitioningI
\ingroup Geometry_classes

Abstract base class for partitioning structures to help reduce the number of
triangles needed to be tested to compute TGeoTessellated's navigation functions.
*/

#ifndef TPARTITIONINGI_HH
#define TPARTITIONINGI_HH

#include "TObject.h"
#include "Math/Vector3D.h"

#include "TGeoTriangleMesh.h"

namespace Tessellated {

class TPartitioningI : public TObject {
protected:
   const TGeoTriangleMesh *fMesh{nullptr}; ///!< non owning pointer to triangle mesh

public:
   virtual void SetTriangleMesh(const TGeoTriangleMesh *trianglemesh) { fMesh = trianglemesh; }
   const TGeoTriangleMesh *GetTriangleMesh() const { return fMesh; }

   /// Is point contained by mesh
   /// \param[in] point to be tested for containment in mesh
   /// \return Bool_t indicating containment
   virtual Bool_t IsPointContained(const ROOT::Math::XYZVector &point) const = 0;

   /// Compute minimal distance between point an mesh surface
   /// \param[in] point for which minimum distance is required
   /// \return Double_t indicating minimum distance
   virtual Double_t GetSafetyDistance(const ROOT::Math::XYZVector &point) const = 0;

   /// Find closest triangle to given point
   /// \param[in] point for which closest triangle is required
   /// \return TGeoTriangleMesh::ClosestTriangle_t
   virtual TGeoTriangleMesh::ClosestTriangle_t GetClosestTriangle(const ROOT::Math::XYZVector &point) const = 0;

   /// Find distance from origin to first intersected triangle in direction
   /// \param[in] origin of ray
   /// \param[in] direction of ray
   /// \param[in] isorigininside indicating if origin is inside mesh
   /// \return Double_t indicating distance to first intersected triangle by ray
   virtual Double_t DistanceInDirection(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction,
                                        Bool_t isorigininside) = 0;

   virtual ~TPartitioningI() override {}

   ClassDefOverride(TPartitioningI, 1)
};

}; // namespace Tessellated

#endif /*TPARTITIONINGI_HH*/
