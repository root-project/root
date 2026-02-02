// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TBVH_HH
#define TBVH_HH

#include <memory>  // for unique_ptr
#include <utility> // for pair
#include <vector>  // for vector

#include "Rtypes.h"           // for THashConsistencyHolder, ClassDefOverride
#include "RtypesCore.h"       // for Double_t, Bool_t, Int_t, kFALSE, UInt_t
#include "TPartitioningI.h"   // for TPartitioningI
#include "TGeoTriangle.h"     // for TGeoTriangle
#include "TGeoTriangleMesh.h" // for TGeoTriangleMesh, TGeoTriangleMesh::ClosestTri...
#include "Math/Vector3D.h"    // for ROOT::Math::XYZVector

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>

class TBuffer;
class TClass;
class TMemberInspector;

namespace Tessellated {

class TBVH : public TPartitioningI {
private:
   using Scalar = Double_t;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using BBox = bvh::v2::BBox<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;
   using Ray = bvh::v2::Ray<Scalar, 3>;
   using Quality = bvh::v2::DefaultBuilder<Node>::Quality;

   std::unique_ptr<bvh::v2::Bvh<Node>> fBVH{nullptr}; ///< bounding volume hierarchy structure
   Quality fBVHQuality{Quality::Medium};              ///< quality for bvh building

private:
   void BuildBVH();
   void ResetInternalState();
   const TGeoTriangle *GetIntersectedTriangle(Ray &ray) const;

public:
   TBVH() = default;
   virtual ~TBVH() override = default;

   virtual void SetTriangleMesh(const TGeoTriangleMesh *trianglemesh) override;
   virtual void SetBVHQuality(bvh::v2::DefaultBuilder<Node>::Quality quality);

   virtual Bool_t IsPointContained(const ROOT::Math::XYZVector &point) const override;
   virtual Double_t GetSafetyDistance(const ROOT::Math::XYZVector &point) const override;
   virtual TGeoTriangleMesh::ClosestTriangle_t GetClosestTriangle(const ROOT::Math::XYZVector &point) const override;
   virtual Double_t DistanceInDirection(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction,
                                        Bool_t isorigininside) override;

   virtual void Print(Option_t *opt = nullptr) const override;

   ClassDefOverride(TBVH, 1)
};
}; // namespace Tessellated

#endif /*TBVH_HH*/
