// Author: Andrei Gheata   17/02/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoParallelWorld
\ingroup Geometry_classes
Base class for a flat parallel geometry.

  The parallel geometry can be composed by both normal volumes added
using the AddNode interface (not implemented yet) or by physical nodes
which will use as position their actual global matrix with respect to the top
volume of the main geometry.

  All these nodes are added as daughters to the "top" volume of
the parallel world which acts as a navigation helper in this parallel
world. The parallel world has to be closed before calling any navigation
method.
*/

#include "TGeoParallelWorld.h"
#include "TObjString.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoVoxelFinder.h"
#include "TGeoMatrix.h"
#include "TGeoPhysicalNode.h"
#include "TGeoNavigator.h"
#include "TGeoBBox.h"
#include "TGeoVoxelGrid.h"
#include "TStopwatch.h"
#include <iostream>
#include <queue>
#include <functional>
#include <mutex>

// this is for the bvh acceleration stuff
#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wall"
#  pragma GCC diagnostic ignored "-Wshadow"
#  pragma GCC diagnostic ignored "-Wunknown-pragmas"
#  pragma GCC diagnostic ignored "-Wattributes"
#elif defined (_MSC_VER)
#  pragma warning( push )
#  pragma warning( disable : 5051)
#endif

// V2 BVH
#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/default_builder.h>

ClassImp(TGeoParallelWorld);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TGeoParallelWorld::TGeoParallelWorld(const char *name, TGeoManager *mgr)
   : TNamed(name, ""),
     fGeoManager(mgr),
     fPaths(new TObjArray(256)),
     fUseOverlaps(kFALSE),
     fIsClosed(kFALSE),
     fVolume(nullptr),
     fLastState(nullptr),
     fPhysical(new TObjArray(256))
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TGeoParallelWorld::~TGeoParallelWorld()
{
   if (fPhysical) {
      fPhysical->Delete();
      delete fPhysical;
   }
   if (fPaths) {
      fPaths->Delete();
      delete fPaths;
   }
   delete fVolume;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a node normally to this world. Overlapping nodes not allowed

void TGeoParallelWorld::AddNode(const char *path)
{
   if (fIsClosed)
      Fatal("AddNode", "Cannot add nodes to a closed parallel geometry");
   if (!fGeoManager->CheckPath(path)) {
      Error("AddNode", "Path %s not valid.\nCannot add to parallel world!", path);
      return;
   }
   fPaths->Add(new TObjString(path));
}

////////////////////////////////////////////////////////////////////////////////
/// To use this optimization, the user should declare the full list of volumes
/// which may overlap with any of the physical nodes of the parallel world. Better
/// be done before misalignment

void TGeoParallelWorld::AddOverlap(TGeoVolume *vol, Bool_t activate)
{
   if (activate)
      fUseOverlaps = kTRUE;
   vol->SetOverlappingCandidate(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// To use this optimization, the user should declare the full list of volumes
/// which may overlap with any of the physical nodes of the parallel world. Better
/// be done before misalignment

void TGeoParallelWorld::AddOverlap(const char *volname, Bool_t activate)
{
   if (activate)
      fUseOverlaps = kTRUE;
   TIter next(fGeoManager->GetListOfVolumes());
   TGeoVolume *vol;
   while ((vol = (TGeoVolume *)next())) {
      if (!strcmp(vol->GetName(), volname))
         vol->SetOverlappingCandidate(kTRUE);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Print the overlaps which were detected during real tracking

Int_t TGeoParallelWorld::PrintDetectedOverlaps() const
{
   TIter next(fGeoManager->GetListOfVolumes());
   TGeoVolume *vol;
   Int_t noverlaps = 0;
   while ((vol = (TGeoVolume *)next())) {
      if (vol->IsOverlappingCandidate()) {
         if (noverlaps == 0)
            Info("PrintDetectedOverlaps", "List of detected volumes overlapping with the PW");
         noverlaps++;
         printf("volume: %s at index: %d\n", vol->GetName(), vol->GetNumber());
      }
   }
   return noverlaps;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset overlapflag for all volumes in geometry

void TGeoParallelWorld::ResetOverlaps() const
{
   TIter next(fGeoManager->GetListOfVolumes());
   TGeoVolume *vol;
   while ((vol = (TGeoVolume *)next()))
      vol->SetOverlappingCandidate(kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// The main geometry must be closed.

Bool_t TGeoParallelWorld::CloseGeometry()
{
   if (fIsClosed)
      return kTRUE;
   if (!fGeoManager->IsClosed())
      Fatal("CloseGeometry", "Main geometry must be closed first");
   if (!fPaths || !fPaths->GetEntriesFast()) {
      Error("CloseGeometry", "List of paths is empty");
      return kFALSE;
   }
   RefreshPhysicalNodes();
   fIsClosed = kTRUE;
   Info("CloseGeometry", "Parallel world %s contains %d prioritised objects", GetName(), fPaths->GetEntriesFast());
   Int_t novlp = 0;
   TIter next(fGeoManager->GetListOfVolumes());
   TGeoVolume *vol;
   while ((vol = (TGeoVolume *)next()))
      if (vol->IsOverlappingCandidate())
         novlp++;
   Info("CloseGeometry", "Number of declared overlaps: %d", novlp);
   if (fUseOverlaps)
      Info("CloseGeometry", "Parallel world will use declared overlaps");
   else
      Info("CloseGeometry", "Parallel world will detect overlaps with other volumes");

   Info("CloseGeometry", "Parallel world has %d volumes", fVolume->GetNdaughters());
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Refresh the node pointers and re-voxelize. To be called mandatory in case
/// re-alignment happened.

void TGeoParallelWorld::RefreshPhysicalNodes()
{
   delete fVolume;
   fVolume = new TGeoVolumeAssembly(GetName());
   fGeoManager->GetListOfVolumes()->Remove(fVolume);
   // Loop physical nodes and add them to the navigation helper volume
   if (fPhysical) {
      fPhysical->Delete();
      delete fPhysical;
   }
   fPhysical = new TObjArray(fPaths->GetEntriesFast());
   TGeoPhysicalNode *pnode;
   TObjString *objs;
   TIter next(fPaths);
   Int_t copy = 0;
   while ((objs = (TObjString *)next())) {
      pnode = new TGeoPhysicalNode(objs->GetName());
      fPhysical->AddAt(pnode, copy);
      fVolume->AddNode(pnode->GetVolume(), copy++, new TGeoHMatrix(*pnode->GetMatrix()));
   }
   // Voxelize the volume
   fVolume->GetShape()->ComputeBBox();
   TStopwatch timer;
   timer.Start();
   auto verboselevel = TGeoManager::GetVerboseLevel();
   if (fAccMode == AccelerationMode::kBVH) {
      this->BuildBVH();
      timer.Stop();
      if (verboselevel > 2) {
         Info("RefreshPhysicalNodes", "Initializing BVH took %f seconds", timer.RealTime());
      }
   }
   if (fAccMode == AccelerationMode::kVoxelFinder) {
      timer.Start();
      fVolume->Voxelize("ALL");
      timer.Stop();
      if (verboselevel > 2) {
         Info("RefreshPhysicalNodes", "Voxelization took %f seconds", timer.RealTime());
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Finds physical node containing the point.
/// Uses BVH to do so. (Not the best algorithm since not O(1) but good enough.)
/// An improved version could be implemented based on TGeoVoxelGrid caching.

TGeoPhysicalNode *TGeoParallelWorld::FindNodeBVH(Double_t point[3])
{
   if (!fIsClosed) {
      Fatal("FindNode", "Parallel geometry must be closed first");
   }

   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;

   // let's fetch the bvh
   auto mybvh = (Bvh *)fBVH;
   assert(mybvh);

   Vec3 testpoint(point[0], point[1], point[2]);

   TGeoPhysicalNode *nextnode = nullptr;

   // This index stores the smallest object_id that contains the point
   // only relevant if there are overlaps within the parallel world.
   // We introduce this to make sure that the BVH traversal here, gives the same
   // result as a simple loop iteration in increasing object_id order.
   size_t min_contained_object_id = std::numeric_limits<size_t>::max();

   auto contains = [](const Node &node, const Vec3 &p) {
      auto box = node.get_bbox();
      auto min = box.min;
      auto max = box.max;
      return (p[0] >= min[0] && p[0] <= max[0]) && (p[1] >= min[1] && p[1] <= max[1]) &&
             (p[2] >= min[2] && p[2] <= max[2]);
   };

   auto leaf_fn = [&](size_t begin, size_t end) {
      for (size_t prim_id = begin; prim_id < end; ++prim_id) {
         auto objectid = mybvh->prim_ids[prim_id];
         if (min_contained_object_id == std::numeric_limits<size_t>::max() || objectid < min_contained_object_id) {
            auto object = fVolume->GetNode(objectid);
            Double_t lpoint[3] = {0};
            object->MasterToLocal(point, lpoint);
            if (object->GetVolume()->Contains(lpoint)) {
               nextnode = (TGeoPhysicalNode *)fPhysical->At(objectid);
               min_contained_object_id = objectid;
            }
         }
      }
      return false; // false == go on with search even after this leaf
   };

   auto root = mybvh->nodes[0];
   // quick check against the root node
   if (!contains(root, testpoint)) {
      nextnode = nullptr;
   } else {
      bvh::v2::GrowingStack<Bvh::Index> stack;
      constexpr bool earlyExit = false; // needed in overlapping cases, in which we prioritize smaller object indices
      mybvh->traverse_top_down<earlyExit>(root.index, stack, leaf_fn, [&](const Node &left, const Node &right) {
         bool follow_left = contains(left, testpoint);
         bool follow_right = contains(right, testpoint);
         return std::make_tuple(follow_left, follow_right, false);
      });
   }

   if (nextnode) {
      fLastState = nextnode;
   }
   return nextnode;
}

////////////////////////////////////////////////////////////////////////////////
/// Finds physical node containing the point
/// (original version based on TGeoVoxelFinder)

TGeoPhysicalNode *TGeoParallelWorld::FindNodeOrig(Double_t point[3])
{
   if (!fIsClosed)
      Fatal("FindNode", "Parallel geometry must be closed first");
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if not in an overlapping candidate
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   Int_t id;
   Int_t ncheck = 0;
   Int_t nd = fVolume->GetNdaughters();
   // get the list of nodes passing thorough the current voxel
   TGeoNodeCache *cache = nav->GetCache();
   TGeoStateInfo &info = *cache->GetMakePWInfo(nd);
   Int_t *check_list = voxels->GetCheckList(point, ncheck, info);
   //   cache->ReleaseInfo(); // no hierarchical use
   if (!check_list)
      return nullptr;
   // loop all nodes in voxel
   TGeoNode *node;
   Double_t local[3];
   for (id = 0; id < ncheck; id++) {
      node = fVolume->GetNode(check_list[id]);
      node->MasterToLocal(point, local);
      if (node->GetVolume()->Contains(local)) {
         // We found a node containing the point
         fLastState = (TGeoPhysicalNode *)fPhysical->At(node->GetNumber());
         return fLastState;
      }
   }
   return nullptr;
}

///////////////////////////////////////////////////////////////////////////////////
/// Finds physical node containing the point using simple algorithm (for debugging)

TGeoPhysicalNode *TGeoParallelWorld::FindNodeLoop(Double_t point[3])
{
   if (!fIsClosed)
      Fatal("FindNode", "Parallel geometry must be closed first");
   Int_t nd = fVolume->GetNdaughters();
   for (int id = 0; id < nd; id++) {
      Double_t local[3] = {0};
      auto node = fVolume->GetNode(id);
      node->MasterToLocal(point, local);
      if (node->GetVolume()->Contains(local)) {
         // We found a node containing the point
         fLastState = (TGeoPhysicalNode *)fPhysical->At(node->GetNumber());
         return fLastState;
      }
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Prints the BVH

void TGeoParallelWorld::PrintBVH() const
{
   using Scalar = float;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;

   // let's fetch the bvh
   auto mybvh = (Bvh *)fBVH;

   for (size_t i = 0; i < mybvh->nodes.size(); ++i) {
      const auto &n = mybvh->nodes[i];
      auto bbox = n.get_bbox();
      auto min = bbox.min;
      auto max = bbox.max;
      long objectid = -1;
      if (n.index.prim_count() > 0) {
         objectid = mybvh->prim_ids[n.index.first_id()];
      }
      std::cout << "NODE id" << i << " "
                << " prim_count: " << n.index.prim_count() << " first_id " << n.index.first_id() << " object_id "
                << objectid << " ( " << min[0] << " , " << min[1] << " , " << min[2] << ")"
                << " ( " << max[0] << " , " << max[1] << " , " << max[2] << ")"
                << "\n";
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Same functionality as TGeoNavigator::FindNextDaughterBoundary for the
/// parallel world. Uses BVH to do so.

TGeoPhysicalNode *
TGeoParallelWorld::FindNextBoundaryBVH(Double_t point[3], Double_t dir[3], Double_t &step, Double_t stepmax)
{
   if (!fIsClosed) {
      Fatal("FindNextBoundary", "Parallel geometry must be closed first");
   }

   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) {
      return nullptr;
   }
   // last touched physical node in the parallel geometry
   if (fLastState && fLastState->IsMatchingState(nav)) {
      return nullptr;
   }

   double local_step = stepmax; // we need this otherwise the lambda get's confused

   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;
   using Ray = bvh::v2::Ray<Scalar, 3>;

   // let's fetch the bvh
   auto mybvh = (Bvh *)fBVH;
   if (!mybvh) {
      Error("FindNextBoundary", "Cannot perform safety; No BVH initialized");
      return nullptr;
   }

   auto truncate_roundup = [](double orig) {
      float epsilon = std::numeric_limits<float>::epsilon() * std::fabs(orig);
      // Add the bias to x before assigning it to y
      return static_cast<float>(orig + epsilon);
   };

   // let's do very quick checks against the top node
   const auto topnode_bbox = mybvh->get_root().get_bbox();
   if ((-point[0] + topnode_bbox.min[0]) > stepmax) {
      step = stepmax;
      return nullptr;
   }
   if ((-point[1] + topnode_bbox.min[1]) > stepmax) {
      step = stepmax;
      return nullptr;
   }
   if ((-point[2] + topnode_bbox.min[2]) > stepmax) {
      step = stepmax;
      return nullptr;
   }
   if ((point[0] - topnode_bbox.max[0]) > stepmax) {
      step = stepmax;
      return nullptr;
   }
   if ((point[1] - topnode_bbox.max[1]) > stepmax) {
      step = stepmax;
      return nullptr;
   }
   if ((point[2] - topnode_bbox.max[2]) > stepmax) {
      step = stepmax;
      return nullptr;
   }

   // the ray used for bvh interaction
   Ray ray(Vec3(point[0], point[1], point[2]), // origin
           Vec3(dir[0], dir[1], dir[2]),       // direction
           0.0f,                               // minimum distance
           truncate_roundup(local_step));

   TGeoPhysicalNode *nextnode = nullptr;

   static constexpr bool use_robust_traversal = true;

   // Traverse the BVH and apply concrecte object intersection in BVH leafs
   bvh::v2::GrowingStack<Bvh::Index> stack;
   mybvh->intersect<false, use_robust_traversal>(ray, mybvh->get_root().index, stack, [&](size_t begin, size_t end) {
      for (size_t prim_id = begin; prim_id < end; ++prim_id) {
         auto objectid = mybvh->prim_ids[prim_id];
         auto object = fVolume->GetNode(objectid);

         auto pnode = (TGeoPhysicalNode *)fPhysical->At(objectid);
         if (pnode->IsMatchingState(nav)) {
            // Info("FOO", "Matching state return");
            step = TGeoShape::Big();
            nextnode = nullptr;
            return true;
         }
         Double_t lpoint[3], ldir[3];
         object->MasterToLocal(point, lpoint);
         object->MasterToLocalVect(dir, ldir);
         auto thisstep = object->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, local_step);
         if (thisstep < local_step) {
            local_step = thisstep;
            nextnode = pnode;
         }
      }
      return false; // go on after this
   });

   // nothing hit
   if (!nextnode) {
      local_step = TGeoShape::Big();
   }
   step = local_step;
   return nextnode;
}

////////////////////////////////////////////////////////////////////////////////
/// Same functionality as TGeoNavigator::FindNextDaughterBoundary for the
/// parallel world.
/// (original version based on TGeoVoxelFinder)

TGeoPhysicalNode *
TGeoParallelWorld::FindNextBoundaryOrig(Double_t point[3], Double_t dir[3], Double_t &step, Double_t stepmax)
{
   if (!fIsClosed)
      Fatal("FindNextBoundary", "Parallel geometry must be closed first");
   TGeoPhysicalNode *pnode = nullptr;
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate())
      return nullptr;
   //   TIter next(fPhysical);
   // Ignore the request if the current state in the main geometry matches the
   // last touched physical node in the parallel geometry
   if (fLastState && fLastState->IsMatchingState(nav))
      return nullptr;
   //   while ((pnode = (TGeoPhysicalNode*)next())) {
   //      if (pnode->IsMatchingState(nav)) return 0;
   //   }
   step = stepmax;
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   Int_t idaughter = -1; // nothing crossed
   Int_t nd = fVolume->GetNdaughters();
   Int_t i;
   TGeoNode *current;
   Double_t lpoint[3], ldir[3];
   //   const Double_t tolerance = TGeoShape::Tolerance();
   if (nd < 5) {
      // loop over daughters
      for (i = 0; i < nd; i++) {
         current = fVolume->GetNode(i);
         pnode = (TGeoPhysicalNode *)fPhysical->At(i);
         if (pnode->IsMatchingState(nav)) {
            step = TGeoShape::Big();
            return nullptr;
         }
         // validate only within stepmax
         if (voxels->IsSafeVoxel(point, i, stepmax))
            continue;
         current->MasterToLocal(point, lpoint);
         current->MasterToLocalVect(dir, ldir);
         Double_t snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, step);
         if (snext < step) {
            step = snext;
            idaughter = i;
         }
      }
      if (idaughter >= 0) {
         pnode = (TGeoPhysicalNode *)fPhysical->At(idaughter);
         return pnode;
      }
      step = TGeoShape::Big();
      return nullptr;
   }
   // Get current voxel
   Int_t ncheck = 0;
   Int_t sumchecked = 0;
   Int_t *vlist = nullptr;
   TGeoNodeCache *cache = nav->GetCache();
   TGeoStateInfo &info = *cache->GetMakePWInfo(nd);
   //   TGeoStateInfo &info = *cache->GetInfo();
   //   cache->ReleaseInfo(); // no hierarchical use
   voxels->SortCrossedVoxels(point, dir, info);
   while ((sumchecked < nd) && (vlist = voxels->GetNextVoxel(point, dir, ncheck, info))) {
      for (i = 0; i < ncheck; i++) {
         pnode = (TGeoPhysicalNode *)fPhysical->At(vlist[i]);
         if (pnode->IsMatchingState(nav)) {
            step = TGeoShape::Big();
            return nullptr;
         }
         current = fVolume->GetNode(vlist[i]);
         current->MasterToLocal(point, lpoint);
         current->MasterToLocalVect(dir, ldir);
         Double_t snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, step);
         if (snext < step - 1.E-8) {
            step = snext;
            idaughter = vlist[i];
         }
      }
      if (idaughter >= 0) {
         pnode = (TGeoPhysicalNode *)fPhysical->At(idaughter);
         // mark the overlap
         if (!fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) {
            AddOverlap(nav->GetCurrentVolume(), kFALSE);
            //            printf("object %s overlapping with pn: %s\n", fGeoManager->GetPath(), pnode->GetName());
         }
         return pnode;
      }
   }
   step = TGeoShape::Big();
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Same functionality as TGeoNavigator::FindNextDaughterBoundary for the
/// parallel world in a trivial loop version (for debugging)

TGeoPhysicalNode *
TGeoParallelWorld::FindNextBoundaryLoop(Double_t point[3], Double_t dir[3], Double_t &step, Double_t stepmax)
{
   if (!fIsClosed) {
      Fatal("FindNextBoundary", "Parallel geometry must be closed first");
   }

   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) {
      return nullptr;
   }
   // last touched physical node in the parallel geometry
   if (fLastState && fLastState->IsMatchingState(nav)) {
      return nullptr;
   }

   step = stepmax;
   int nd = fVolume->GetNdaughters();
   TGeoPhysicalNode *nextnode = nullptr;
   for (int i = 0; i < nd; ++i) {
      auto object = fVolume->GetNode(i);
      // check actual distance/safety to object
      Double_t lpoint[3], ldir[3];

      object->MasterToLocal(point, lpoint);
      object->MasterToLocalVect(dir, ldir);
      auto thisstep = object->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, step);
      if (thisstep < step) {
         step = thisstep;
         nextnode = (TGeoPhysicalNode *)fPhysical->At(i);
      }
   }
   // nothing hit
   if (!nextnode) {
      step = TGeoShape::Big();
   }
   return nextnode;
}

namespace {
// some helpers for point - axis aligned bounding box functions
// using bvh types

// determines if a point is inside the bounding box
template <typename T>
bool contains(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   auto min = box.min;
   auto max = box.max;
   return (p[0] >= min[0] && p[0] <= max[0]) && (p[1] >= min[1] && p[1] <= max[1]) &&
          (p[2] >= min[2] && p[2] <= max[2]);
}

// determines the largest squared distance of point to any of the bounding box corners
template <typename T>
auto RmaxSqToNode(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   // construct the 8 corners to get the maximal distance
   const auto minCorner = box.min;
   const auto maxCorner = box.max;
   using Vec3 = bvh::v2::Vec<T, 3>;
   // these are the corners of the bounding box
   const std::array<bvh::v2::Vec<T, 3>, 8> corners{
      Vec3{minCorner[0], minCorner[1], minCorner[2]}, Vec3{minCorner[0], minCorner[1], maxCorner[2]},
      Vec3{minCorner[0], maxCorner[1], minCorner[2]}, Vec3{minCorner[0], maxCorner[1], maxCorner[2]},
      Vec3{maxCorner[0], minCorner[1], minCorner[2]}, Vec3{maxCorner[0], minCorner[1], maxCorner[2]},
      Vec3{maxCorner[0], maxCorner[1], minCorner[2]}, Vec3{maxCorner[0], maxCorner[1], maxCorner[2]}};

   T Rmax_sq{0};
   for (const auto &corner : corners) {
      float R_sq = 0.;
      const auto dx = corner[0] - p[0];
      R_sq += dx * dx;
      const auto dy = corner[1] - p[1];
      R_sq += dy * dy;
      const auto dz = corner[2] - p[2];
      R_sq += dz * dz;
      Rmax_sq = std::max(Rmax_sq, R_sq);
   }
   return Rmax_sq;
};

// determines the mininum squared distance of point to a bounding box ("safey square")
template <typename T>
auto SafetySqToNode(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   T sqDist{0.0};
   for (int i = 0; i < 3; i++) {
      T v = p[i];
      if (v < box.min[i]) {
         sqDist += (box.min[i] - v) * (box.min[i] - v);
      } else if (v > box.max[i]) {
         sqDist += (v - box.max[i]) * (v - box.max[i]);
      }
   }
   return sqDist;
};

// Helper classes/structs used for priority queue - BVH traversal

// structure keeping cost (value) for a BVH index
struct BVHPrioElement {
   size_t bvh_node_id;
   float value;
};

// A priority queue for BVHPrioElement with an additional clear method
// for quick reset
template <typename Comparator>
class BVHPrioQueue : public std::priority_queue<BVHPrioElement, std::vector<BVHPrioElement>, Comparator> {
public:
   using std::priority_queue<BVHPrioElement, std::vector<BVHPrioElement>,
                             Comparator>::priority_queue; // constructor inclusion

   // convenience method to quickly clear/reset the queue (instead of having to pop one by one)
   void clear() { this->c.clear(); }
};

} // namespace

////////////////////////////////////////////////////////////////////////////////////////
/// Method to find potentially relevant candidate bounding boxes for safety calculation
/// given a point. Uses trivial algorithm to do so.

std::pair<double, double>
TGeoParallelWorld::GetLoopSafetyCandidates(double point[3], std::vector<int> &candidates, double margin) const
{
   // Given a 3D point, returns the
   // a) the min radius R such that there is at least one leaf bounding box fully enclosed
   //    in the sphere of radius R around point + the smallest squared safety
   // b) the set of leaf bounding boxes who partially lie within radius + margin

   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using BBox = bvh::v2::BBox<Scalar, 3>;
   const auto bboxes_ptr = (std::vector<BBox> *)fBoundingBoxes;
   auto &bboxes = (*bboxes_ptr);

   auto cmp = [](BVHPrioElement a, BVHPrioElement b) { return a.value > b.value; };
   static thread_local BVHPrioQueue<decltype(cmp)> queue(cmp);
   queue.clear();

   // testpoint object in float for quick BVH interaction
   Vec3 testpoint(point[0], point[1], point[2]);
   float best_enclosing_R_sq = std::numeric_limits<float>::max();
   for (size_t i = 0; i < bboxes.size(); ++i) {
      const auto &thisbox = bboxes[i];
      auto safety_sq = SafetySqToNode(thisbox, testpoint);
      const auto this_R_max_sq = RmaxSqToNode(thisbox, testpoint);
      const auto inside = contains(thisbox, testpoint);
      safety_sq = inside ? -1.f : safety_sq;
      best_enclosing_R_sq = std::min(best_enclosing_R_sq, this_R_max_sq);
      queue.emplace(BVHPrioElement{i, safety_sq});
   }

   // now we know the best enclosing R
   // **and** we can fill the candidate set from the priority queue easily
   float safety_sq_at_least = -1.f;

   // final transform in order to take into account margin
   if (margin != 0.) {
      float best_enclosing_R = std::sqrt(best_enclosing_R_sq) + margin;
      best_enclosing_R_sq = best_enclosing_R * best_enclosing_R;
   }

   if (queue.size() > 0) {
      auto el = queue.top();
      queue.pop();
      safety_sq_at_least = el.value; // safety_sq;
      while (el.value /*safety_sq*/ < best_enclosing_R_sq) {
         candidates.emplace_back(el.bvh_node_id);
         if (queue.size() > 0) {
            el = queue.top();
            queue.pop();
         } else {
            break;
         }
      }
   }
   return std::make_pair<double, double>(best_enclosing_R_sq, safety_sq_at_least);
}

////////////////////////////////////////////////////////////////////////////////////////
/// Method to find potentially relevant candidate bounding boxes for safety calculation
/// given a point. Uses BVH to do so.

std::pair<double, double>
TGeoParallelWorld::GetBVHSafetyCandidates(double point[3], std::vector<int> &candidates, double margin) const
{
   // Given a 3D point, returns the
   // a) the min radius R such that there is at least one leaf bounding box fully enclosed
   //    in the sphere of radius R around point
   // b) the set of leaf bounding boxes who partially lie within radius + margin

   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;
   using BBox = bvh::v2::BBox<Scalar, 3>;
   // let's fetch the primitive bounding boxes
   const auto bboxes = (std::vector<BBox> *)fBoundingBoxes;
   // let's fetch the bvh
   auto mybvh = (Bvh *)fBVH;

   // testpoint object in float for quick BVH interaction
   Vec3 testpoint(point[0], point[1], point[2]);

   // comparator bringing out "smallest" value on top
   auto cmp = [](BVHPrioElement a, BVHPrioElement b) { return a.value > b.value; };
   static thread_local BVHPrioQueue<decltype(cmp)> queue(cmp);
   queue.clear();
   static thread_local BVHPrioQueue<decltype(cmp)> leaf_queue(cmp);
   leaf_queue.clear();

   auto currnode = mybvh->nodes[0]; // we start from the top BVH node
   // algorithm is based on standard iterative tree traversal with priority queues
   float best_enclosing_R_sq = std::numeric_limits<float>::max();
   float best_enclosing_R_with_margin_sq = std::numeric_limits<float>::max();
   float current_safety_sq = 0.f;
   do {
      if (currnode.is_leaf()) {
         // we are in a leaf node and actually talk to primitives
         const auto begin_prim_id = currnode.index.first_id();
         const auto end_prim_id = begin_prim_id + currnode.index.prim_count();
         for (auto p_id = begin_prim_id; p_id < end_prim_id; p_id++) {
            const auto object_id = mybvh->prim_ids[p_id];
            //
            // fetch leaf_bounding box
            const auto &leaf_bounding_box = (*bboxes)[object_id];
            auto this_Rmax_sq = RmaxSqToNode(leaf_bounding_box, testpoint);
            const bool inside = contains(leaf_bounding_box, testpoint);
            const auto safety_sq = inside ? -1.f : SafetySqToNode(leaf_bounding_box, testpoint);

            // update best Rmin
            if (this_Rmax_sq < best_enclosing_R_sq) {
               best_enclosing_R_sq = this_Rmax_sq;
               const auto this_Rmax = std::sqrt(this_Rmax_sq);
               best_enclosing_R_with_margin_sq = (this_Rmax + margin) * (this_Rmax + margin);
            }

            // best_enclosing_R_sq = std::min(best_enclosing_R_sq, this_Rmax_sq);
            if (safety_sq <= best_enclosing_R_with_margin_sq) {
               // update the priority queue of leaf bounding boxes
               leaf_queue.emplace(BVHPrioElement{object_id, safety_sq});
            }
         }
      } else {
         // not a leave node ... for further traversal,
         // we inject the children into priority queue based on distance to it's bounding box
         const auto leftchild_id = currnode.index.first_id();
         const auto rightchild_id = leftchild_id + 1;

         for (size_t childid : {leftchild_id, rightchild_id}) {
            if (childid >= mybvh->nodes.size()) {
               continue;
            }
            const auto &node = mybvh->nodes[childid];
            const auto &thisbbox = node.get_bbox();
            auto inside = contains(thisbbox, testpoint);
            const auto this_safety_sq = inside ? -1.f : SafetySqToNode(thisbbox, testpoint);
            if (this_safety_sq <= best_enclosing_R_with_margin_sq) {
               // this should be further considered
               queue.push(BVHPrioElement{childid, this_safety_sq});
            }
         }
      }

      if (queue.size() > 0) {
         auto currElement = queue.top();
         currnode = mybvh->nodes[currElement.bvh_node_id];
         current_safety_sq = currElement.value;
         queue.pop();
      } else {
         break;
      }
   } while (current_safety_sq <= best_enclosing_R_with_margin_sq);

   // now we know the best enclosing R
   // **and** we can fill the candidate set from the leaf priority queue easily
   float safety_sq_at_least = -1.f;
   if (leaf_queue.size() > 0) {
      auto el = leaf_queue.top();
      leaf_queue.pop();
      safety_sq_at_least = el.value;
      while (el.value < best_enclosing_R_with_margin_sq) {
         candidates.emplace_back(el.bvh_node_id);
         if (leaf_queue.size() > 0) {
            el = leaf_queue.top();
            leaf_queue.pop();
         } else {
            break;
         }
      }
   }
   return std::make_pair<double, double>(best_enclosing_R_sq, safety_sq_at_least);
}

////////////////////////////////////////////////////////////////////////////////
/// Method to initialize the safety voxel at a specific 3D voxel (grid) index
///

void TGeoParallelWorld::InitSafetyVoxel(TGeoVoxelGridIndex const &vi)
{
   static std::mutex g_mutex;
   // this function modifies cache state ---> make writing thread-safe
   const std::lock_guard<std::mutex> lock(g_mutex);

   // determine voxel midpoint
   const auto [mpx, mpy, mpz] = fSafetyVoxelCache->getVoxelMidpoint(vi);
   static std::vector<int> candidates;
   candidates.clear();
   double point[3] = {mpx, mpy, mpz};
   auto [encl_Rmax_sq, min_safety_sq] =
      GetBVHSafetyCandidates(point, candidates, fSafetyVoxelCache->getDiagonalLength());

   // cache information
   auto voxelinfo = fSafetyVoxelCache->at(vi);
   voxelinfo->min_safety = std::sqrt(min_safety_sq);
   voxelinfo->idx_start = fSafetyCandidateStore.size();
   voxelinfo->num_candidates = candidates.size();

   // update flat candidate store
   std::copy(candidates.begin(), candidates.end(), std::back_inserter(fSafetyCandidateStore));
}

////////////////////////////////////////////////////////////////////////////////
/// Compute safety for the parallel world
/// used BVH structure with addiditional on-the-fly 3D grid/voxel caching ---> essentially an O(1) algorithm !)

double TGeoParallelWorld::VoxelSafety(double point[3], double safe_max)
{
   // (a): Very fast checks against top box and global caching
   // (b): Use voxel cached best safety
   // (c): Use voxel candidates (fetch them if they are not yet initialized)

   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if the state matches the last one recorded

   if (fLastState && fLastState->IsMatchingState(nav)) {
      return TGeoShape::Big();
   }

   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) {
      return TGeoShape::Big();
   }

   // let's determine the voxel indices
   TGeoVoxelGridIndex vi = fSafetyVoxelCache->pointToVoxelIndex((float)point[0], (float)point[1], (float)point[2]);
   if (!vi.isValid()) {
      return SafetyBVH(point, safe_max);
   }

   auto &voxelinfo = fSafetyVoxelCache->fGrid[vi.idx];
   double bestsafety = safe_max;

   if (!voxelinfo.isInitialized()) {
      // initialize the cache at this voxel
      InitSafetyVoxel(vi);
   }

   if (voxelinfo.min_safety > 0) {
      // Nothing to do if this is already much further away than safe_max (within the margin limits of the voxel)
      if (voxelinfo.min_safety - fSafetyVoxelCache->getDiagonalLength() > safe_max) {
         return safe_max;
      }

      // see if the precalculated (mid-point) safety value can be used
      auto midpoint = fSafetyVoxelCache->getVoxelMidpoint(vi);
      double r_sq = 0;
      for (int i = 0; i < 3; ++i) {
         const auto d = (point[i] - midpoint[i]);
         r_sq += d * d;
      }
      if (r_sq < voxelinfo.min_safety * voxelinfo.min_safety) {
         // std::cout << " Still within cached safety ... remaining safety would be " << ls_eval - std::sqrt(r) << "\n";
         return voxelinfo.min_safety - std::sqrt(r_sq);
      }
   }

   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using BBox = bvh::v2::BBox<Scalar, 3>;
   const auto bboxes_ptr = (std::vector<BBox> *)fBoundingBoxes;
   auto &bboxes = (*bboxes_ptr);
   Vec3 testpoint(point[0], point[1], point[2]);
   // do the full calculation using all candidates here
   for (size_t store_id = voxelinfo.idx_start; store_id < voxelinfo.idx_start + voxelinfo.num_candidates; ++store_id) {

      const auto cand_id = fSafetyCandidateStore[store_id];

      // check against bounding box
      const auto &bbox = bboxes[cand_id];
      const auto bbox_safe_sq = SafetySqToNode(bbox, testpoint);
      if (bbox_safe_sq > bestsafety * bestsafety) {
         continue;
      }

      // check against actual geometry primitive-safety
      const auto primitive_node = fVolume->GetNode(cand_id);
      const auto thissafety = primitive_node->Safety(point, false);
      if (thissafety < bestsafety) {
         bestsafety = std::max(0., thissafety);
      }
   }
   return bestsafety;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute safety for the parallel world
/// (using pure BVH traversal, mainly for debugging/fallback since VoxelSafety should be faster)

double TGeoParallelWorld::SafetyBVH(double point[3], double safe_max)
{
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if the state matches the last one recorded
   if (fLastState && fLastState->IsMatchingState(nav)) {
      return TGeoShape::Big();
   }
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) {
      return TGeoShape::Big();
   }

   float smallest_safety = safe_max;
   float smallest_safety_sq = smallest_safety * smallest_safety;

   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;

   // let's fetch the bvh
   auto mybvh = (Bvh *)fBVH;

   // testpoint object in float for quick BVH interaction
   Vec3 testpoint(point[0], point[1], point[2]);

   auto currnode = mybvh->nodes[0]; // we start from the top BVH node
   // we do a quick check on the top node
   bool outside_top = !::contains(currnode.get_bbox(), testpoint);
   const auto safety_sq_to_top = SafetySqToNode(currnode.get_bbox(), testpoint);
   if (outside_top && safety_sq_to_top > smallest_safety_sq) {
      // the top node is already further away than our limit, so we can simply return the limit
      return smallest_safety;
   }

   // comparator bringing out "smallest" value on top
   auto cmp = [](BVHPrioElement a, BVHPrioElement b) { return a.value > b.value; };
   static thread_local BVHPrioQueue<decltype(cmp)> queue(cmp);
   queue.clear();

   // algorithm is based on standard iterative tree traversal with priority queues
   float current_safety_to_node_sq = outside_top ? safety_sq_to_top : 0.f;
   do {
      if (currnode.is_leaf()) {
         // we are in a leaf node and actually talk to TGeo primitives
         const auto begin_prim_id = currnode.index.first_id();
         const auto end_prim_id = begin_prim_id + currnode.index.prim_count();

         for (auto p_id = begin_prim_id; p_id < end_prim_id; p_id++) {
            const auto object_id = mybvh->prim_ids[p_id];

            auto pnode = (TGeoPhysicalNode *)fPhysical->UncheckedAt(object_id);
            // Return if inside the current node
            if (pnode->IsMatchingState(nav)) {
               return TGeoShape::Big();
            }

            auto object = fVolume->GetNode(object_id);
            // check actual distance/safety to object
            auto thissafety = object->Safety(point, false);

            // this value (if not zero or negative) may be approximative but we know that it can actually not be smaller
            // than the relevant distance to the bounding box of the node!! Hence we can correct it.

            // if (thissafety > 1E-10 && thissafety * thissafety < current_safety_to_node_sq) {
            //   thissafety = std::max(thissafety, double(std::sqrt(current_safety_to_node_sq)));
            // }

            if (thissafety < smallest_safety) {
               smallest_safety = std::max(0., thissafety);
               smallest_safety_sq = smallest_safety * smallest_safety;
            }
         }
      } else {
         // not a leave node ... for further traversal,
         // we inject the children into priority queue based on distance to it's bounding box
         const auto leftchild_id = currnode.index.first_id();
         const auto rightchild_id = leftchild_id + 1;

         for (size_t childid : {leftchild_id, rightchild_id}) {
            if (childid >= mybvh->nodes.size()) {
               continue;
            }

            const auto &node = mybvh->nodes[childid];
            const auto inside = contains(node.get_bbox(), testpoint);

            if (inside) {
               // this must be further considered because we are inside the bounding box
               queue.push(BVHPrioElement{childid, -1.});
            } else {
               auto safety_to_node_square = SafetySqToNode(node.get_bbox(), testpoint);
               if (safety_to_node_square <= smallest_safety_sq) {
                  // this should be further considered
                  queue.push(BVHPrioElement{childid, safety_to_node_square});
               }
            }
         }
      }

      if (queue.size() > 0) {
         auto currElement = queue.top();
         currnode = mybvh->nodes[currElement.bvh_node_id];
         current_safety_to_node_sq = currElement.value;
         queue.pop();
      } else {
         break;
      }
   } while (current_safety_to_node_sq <= smallest_safety_sq);

   const auto s = std::max(0., double(smallest_safety));
   return s;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute safety for the parallel world
/// (original version based on TGeoVoxelFinder)

Double_t TGeoParallelWorld::SafetyOrig(Double_t point[3], Double_t safmax)
{
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if the state matches the last one recorded
   if (fLastState && fLastState->IsMatchingState(nav))
      return TGeoShape::Big();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate())
      return TGeoShape::Big();
   Double_t local[3];
   Double_t safe = safmax;
   Double_t safnext;
   TGeoPhysicalNode *pnode = nullptr;
   const Double_t tolerance = TGeoShape::Tolerance();
   Int_t nd = fVolume->GetNdaughters();
   TGeoNode *current;
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   //---> check fast unsafe voxels
   Double_t *boxes = voxels->GetBoxes();
   for (Int_t id = 0; id < nd; id++) {
      Int_t ist = 6 * id;
      Double_t dxyz = 0.;
      Double_t dxyz0 = TMath::Abs(point[0] - boxes[ist + 3]) - boxes[ist];
      if (dxyz0 > safe)
         continue;
      Double_t dxyz1 = TMath::Abs(point[1] - boxes[ist + 4]) - boxes[ist + 1];
      if (dxyz1 > safe)
         continue;
      Double_t dxyz2 = TMath::Abs(point[2] - boxes[ist + 5]) - boxes[ist + 2];
      if (dxyz2 > safe)
         continue;
      if (dxyz0 > 0)
         dxyz += dxyz0 * dxyz0;
      if (dxyz1 > 0)
         dxyz += dxyz1 * dxyz1;
      if (dxyz2 > 0)
         dxyz += dxyz2 * dxyz2;
      if (dxyz >= safe * safe)
         continue;

      pnode = (TGeoPhysicalNode *)fPhysical->At(id);
      // Return if inside the current node
      if (pnode->IsMatchingState(nav)) {
         return TGeoShape::Big();
      }

      current = fVolume->GetNode(id);
      current->MasterToLocal(point, local);
      // Safety to current node
      safnext = current->GetVolume()->GetShape()->Safety(local, kFALSE);
      if (safnext < tolerance)
         return 0.;
      if (safnext < safe)
         safe = safnext;
   }
   return safe;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute safety for the parallel world
/// (trivial loop version for comparison/debugging)

Double_t TGeoParallelWorld::SafetyLoop(Double_t point[3], Double_t safmax)
{
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if the state matches the last one recorded
   if (fLastState && fLastState->IsMatchingState(nav))
      return TGeoShape::Big();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate())
      return TGeoShape::Big();

   Double_t local[3];
   Double_t safe = safmax;
   Double_t safnext;
   const Double_t tolerance = TGeoShape::Tolerance();
   Int_t nd = fVolume->GetNdaughters();

   for (Int_t id = 0; id < nd; id++) {
      auto current = fVolume->GetNode(id);
      current->MasterToLocal(point, local);
      if (current->GetVolume()->GetShape()->Contains(local)) {
         safnext = 0.;
      } else {
         // Safety to current node
         safnext = current->GetVolume()->GetShape()->Safety(local, kFALSE);
      }
      if (safnext < tolerance) {
         return 0.;
      }
      if (safnext < safe) {
         safe = safnext;
      }
   }
   return safe;
}

////////////////////////////////////////////////////////////////////////////////
/// Check overlaps within a tolerance value.

void TGeoParallelWorld::CheckOverlaps(Double_t ovlp)
{
   fVolume->CheckOverlaps(ovlp);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw the parallel world

void TGeoParallelWorld::Draw(Option_t *option)
{
   fVolume->Draw(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Check/validate the BVH acceleration structure.

bool TGeoParallelWorld::CheckBVH(void *bvh, size_t expected_leaf_count) const
{
   using Scalar = float;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;
   auto mybvh = (Bvh *)bvh;

   size_t leaf_count = 0;
   std::function<bool(Node const &)> checkNode = [&](Node const &nde) -> bool {
      if (nde.is_leaf()) {
         leaf_count += nde.index.prim_count();
         return nde.index.prim_count() > 0;
      }

      // we do it recursively
      auto thisbb = nde.get_bbox();

      auto leftindex = nde.index.first_id();
      auto rightindex = leftindex + 1;

      auto leftnode = mybvh->nodes[leftindex];
      auto rightnode = mybvh->nodes[rightindex];

      auto leftbb = leftnode.get_bbox();
      auto rightbb = rightnode.get_bbox();

      // both of these boxes must be contained in the bounding box
      // of the outer node
      auto tmi = thisbb.min;
      auto lmi = leftbb.min;
      auto rmi = rightbb.min;

      auto check1 = lmi[0] >= tmi[0] && lmi[1] >= tmi[1] && lmi[2] >= tmi[2];
      auto check2 = rmi[0] >= tmi[0] && rmi[1] >= tmi[1] && rmi[2] >= tmi[2];

      auto tma = thisbb.max;
      auto lma = leftbb.max;
      auto rma = rightbb.max;

      auto check3 = lma[0] <= tma[0] && lma[1] <= tma[1] && lma[2] <= tma[2];
      auto check4 = rma[0] <= tma[0] && rma[1] <= tma[1] && rma[2] <= tma[2];

      auto check = check1 && check2 && check3 && check4;

      return checkNode(leftnode) && checkNode(rightnode) && check;
   };

   auto check = checkNode(mybvh->nodes[0]);
   return check && leaf_count == expected_leaf_count;
}

////////////////////////////////////////////////////////////////////////////////
/// Build the BVH acceleration structure.

void TGeoParallelWorld::BuildBVH()
{
   using Scalar = float;
   using BBox = bvh::v2::BBox<Scalar, 3>;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;

   auto DaughterToMother = [](TGeoNode const *node, const Double_t *local, Double_t *master) {
      TGeoMatrix *mat = node->GetMatrix();
      if (!mat) {
         memcpy(master, local, 3 * sizeof(Double_t));
      } else {
         mat->LocalToMaster(local, master);
      }
   };

   // helper determining axis aligned bounding box from TGeoNode; code copied from the TGeoVoxelFinder
   auto GetBoundingBoxInMother = [DaughterToMother](TGeoNode const *node) {
      Double_t vert[24] = {0};
      Double_t pt[3] = {0};
      Double_t xyz[6] = {0};
      TGeoBBox *box = (TGeoBBox *)node->GetVolume()->GetShape();
      box->SetBoxPoints(&vert[0]);
      for (Int_t point = 0; point < 8; point++) {
         DaughterToMother(node, &vert[3 * point], &pt[0]);
         if (!point) {
            xyz[0] = xyz[1] = pt[0];
            xyz[2] = xyz[3] = pt[1];
            xyz[4] = xyz[5] = pt[2];
            continue;
         }
         for (Int_t j = 0; j < 3; j++) {
            if (pt[j] < xyz[2 * j]) {
               xyz[2 * j] = pt[j];
            }
            if (pt[j] > xyz[2 * j + 1]) {
               xyz[2 * j + 1] = pt[j];
            }
         }
      }
      BBox bbox;
      bbox.min[0] = std::min(xyz[1], xyz[0]) - 0.001f;
      bbox.min[1] = std::min(xyz[3], xyz[2]) - 0.001f;
      bbox.min[2] = std::min(xyz[5], xyz[4]) - 0.001f;
      bbox.max[0] = std::max(xyz[0], xyz[1]) + 0.001f;
      bbox.max[1] = std::max(xyz[2], xyz[3]) + 0.001f;
      bbox.max[2] = std::max(xyz[4], xyz[5]) + 0.001f;
      return bbox;
   };

   // we need bounding boxes enclosing the primitives and centers of primitives
   // (replaced here by centers of bounding boxes) to build the bvh
   auto bboxes_ptr = new std::vector<BBox>();
   fBoundingBoxes = (void *)bboxes_ptr;
   auto &bboxes = *bboxes_ptr;
   std::vector<Vec3> centers;

   int nd = fVolume->GetNdaughters();
   for (int i = 0; i < nd; ++i) {
      auto node = fVolume->GetNode(i);
      // fetch the bounding box of this node and add to the vector of bounding boxes
      (bboxes).push_back(GetBoundingBoxInMother(node));
      centers.emplace_back((bboxes).back().get_center());
   }

   // check if some previous object is registered and delete if necessary
   if (fBVH) {
      delete (Bvh *)fBVH;
      fBVH = nullptr;
   }

   // create the bvh
   typename bvh::v2::DefaultBuilder<Node>::Config config;
   config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
   auto bvh = bvh::v2::DefaultBuilder<Node>::build(bboxes, centers, config);
   auto bvhptr = new Bvh;
   *bvhptr = std::move(bvh); // copy structure
   fBVH = (void *)(bvhptr);

   auto check = CheckBVH(fBVH, nd);
   if (!check) {
      Error("BuildBVH", "BVH corrupted\n");
   } else {
      Info("BuildBVH", "BVH good\n");
   }

   // now instantiate the 3D voxel grid for caching the safety candidates
   // (note that the structure is merely reserved ... actual filling will happen on-the-fly later on)
   const auto &topBB = bvhptr->get_root().get_bbox();
   int N = std::cbrt(bboxes.size()) + 1;
   // std::cout << "3D Safety GRID cache: Determined N to be " << N << "\n";
   double Lx = (topBB.max[0] - topBB.min[0]) / N;
   double Ly = (topBB.max[1] - topBB.min[1]) / N;
   double Lz = (topBB.max[2] - topBB.min[2]) / N;
   // TODO: Instead of equal number of voxels in each direction, we
   // could impose equal "cubic" voxel size

   if (fSafetyVoxelCache) {
      delete fSafetyVoxelCache;
      fSafetyCandidateStore.clear();
   }

   fSafetyVoxelCache = new TGeoVoxelGrid<SafetyVoxelInfo>(topBB.min[0], topBB.min[1], topBB.min[2], topBB.max[0],
                                                          topBB.max[1], topBB.max[2], Lx, Ly, Lz);

   // TestVoxelGrid();
   return;
}

void TGeoParallelWorld::TestVoxelGrid()
{
   static bool done = false;
   if (done) {
      return;
   }
   done = true;

   auto NX = fSafetyVoxelCache->getVoxelCountX();
   auto NY = fSafetyVoxelCache->getVoxelCountY();
   auto NZ = fSafetyVoxelCache->getVoxelCountZ();

   std::vector<int> candidates1;
   std::vector<int> candidates2;

   for (int i = 0; i < NX; ++i) {
      for (int j = 0; j < NY; ++j) {
         for (int k = 0; k < NZ; ++k) {
            size_t idx = fSafetyVoxelCache->index(i, j, k);
            TGeoVoxelGridIndex vi{i, j, k, idx};

            // midpoint
            auto mp = fSafetyVoxelCache->getVoxelMidpoint(vi);

            // let's do some tests
            candidates1.clear();
            candidates2.clear();
            double point[3] = {mp[0], mp[1], mp[2]};
            auto [encl_Rmax_sq_1, min_safety_sq_1] =
               GetBVHSafetyCandidates(point, candidates1, fSafetyVoxelCache->getDiagonalLength());
            auto [encl_Rmax_sq_2, min_safety_sq_2] =
               GetLoopSafetyCandidates(point, candidates2, fSafetyVoxelCache->getDiagonalLength());
            if (candidates1.size() != candidates2.size()) {
               std::cerr << " i " << i << " " << j << " " << k << " RMAX 2 (BVH) " << encl_Rmax_sq_1 << " CANDSIZE "
                         << candidates1.size() << " RMAX (LOOP) " << encl_Rmax_sq_2 << " CANDSIZE "
                         << candidates2.size() << "\n";
            }

            // the candidate indices have to be the same
            bool same_test1 = true;
            for (auto &id : candidates1) {
               auto ok = std::find(candidates2.begin(), candidates2.end(), id) != candidates2.end();
               if (!ok) {
                  same_test1 = false;
                  break;
               }
            }
            bool same_test2 = true;
            for (auto &id : candidates2) {
               auto ok = std::find(candidates1.begin(), candidates1.end(), id) != candidates1.end();
               if (!ok) {
                  same_test2 = false;
                  break;
               }
            }
            if (!(same_test1 && same_test2)) {
               Error("VoxelTest", "Same test fails %d %d", same_test1, same_test2);
            }
         }
      }
   }
}

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#elif defined (_MSC_VER)
#  pragma warning( pop )
#endif
