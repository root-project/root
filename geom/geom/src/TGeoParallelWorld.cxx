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

ClassImp(TGeoParallelWorld);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TGeoParallelWorld::TGeoParallelWorld(const char *name, TGeoManager *mgr)
                  : TNamed(name,""),
                    fGeoManager(mgr),
                    fPaths(new TObjArray(256)),
                    fUseOverlaps(kFALSE),
                    fIsClosed(kFALSE),
                    fVolume(0),
                    fLastState(0),
                    fPhysical(new TObjArray(256))
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TGeoParallelWorld::~TGeoParallelWorld()
{
   if (fPhysical) {fPhysical->Delete(); delete fPhysical;}
   if (fPaths) {fPaths->Delete(); delete fPaths;}
   delete fVolume;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a node normally to this world. Overlapping nodes not allowed

void TGeoParallelWorld::AddNode(const char *path)
{
   if (fIsClosed) Fatal("AddNode", "Cannot add nodes to a closed parallel geometry");
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
   if (activate) fUseOverlaps = kTRUE;
   vol->SetOverlappingCandidate(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// To use this optimization, the user should declare the full list of volumes
/// which may overlap with any of the physical nodes of the parallel world. Better
/// be done before misalignment

void TGeoParallelWorld::AddOverlap(const char *volname, Bool_t activate)
{
   if (activate) fUseOverlaps = kTRUE;
   TIter next(fGeoManager->GetListOfVolumes());
   TGeoVolume *vol;
   while ((vol=(TGeoVolume*)next())) {
      if (!strcmp(vol->GetName(), volname)) vol->SetOverlappingCandidate(kTRUE);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Print the overlaps which were detected during real tracking

Int_t TGeoParallelWorld::PrintDetectedOverlaps() const
{
   TIter next(fGeoManager->GetListOfVolumes());
   TGeoVolume *vol;
   Int_t noverlaps = 0;
   while ((vol=(TGeoVolume*)next())) {
      if (vol->IsOverlappingCandidate()) {
         if (noverlaps==0) Info("PrintDetectedOverlaps", "List of detected volumes overlapping with the PW");
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
   while ((vol=(TGeoVolume*)next())) vol->SetOverlappingCandidate(kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// The main geometry must be closed.

Bool_t TGeoParallelWorld::CloseGeometry()
{
   if (fIsClosed) return kTRUE;
   if (!fGeoManager->IsClosed()) Fatal("CloseGeometry", "Main geometry must be closed first");
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
   while ((vol=(TGeoVolume*)next())) if (vol->IsOverlappingCandidate()) novlp++;
   Info("CloseGeometry", "Number of declared overlaps: %d", novlp);
   if (fUseOverlaps) Info("CloseGeometry", "Parallel world will use declared overlaps");
   else              Info("CloseGeometry", "Parallel world will detect overlaps with other volumes");
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
   if (fPhysical) {fPhysical->Delete(); delete fPhysical;}
   fPhysical = new TObjArray(fPaths->GetEntriesFast());
   TGeoPhysicalNode *pnode;
   TObjString *objs;
   TIter next(fPaths);
   Int_t copy = 0;
   while ((objs = (TObjString*)next())) {
      pnode = new TGeoPhysicalNode(objs->GetName());
      fPhysical->AddAt(pnode, copy);
      fVolume->AddNode(pnode->GetVolume(), copy++, new TGeoHMatrix(*pnode->GetMatrix()));
   }
   // Voxelize the volume
   fVolume->GetShape()->ComputeBBox();
   fVolume->Voxelize("ALL");
}

////////////////////////////////////////////////////////////////////////////////
/// Finds physical node containing the point

TGeoPhysicalNode *TGeoParallelWorld::FindNode(Double_t point[3])
{
   if (!fIsClosed) Fatal("FindNode", "Parallel geometry must be closed first");
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
   if (!check_list) return 0;
   // loop all nodes in voxel
   TGeoNode *node;
   Double_t local[3];
   for (id=0; id<ncheck; id++) {
      node = fVolume->GetNode(check_list[id]);
      node->MasterToLocal(point, local);
      if (node->GetVolume()->Contains(local)) {
         // We found a node containing the point
         fLastState = (TGeoPhysicalNode*)fPhysical->At(node->GetNumber());
         return fLastState;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Same functionality as TGeoNavigator::FindNextDaughterBoundary for the
/// parallel world

TGeoPhysicalNode *TGeoParallelWorld::FindNextBoundary(Double_t point[3], Double_t dir[3],
                              Double_t &step, Double_t stepmax)
{
   if (!fIsClosed) Fatal("FindNextBoundary", "Parallel geometry must be closed first");
   TGeoPhysicalNode *pnode = 0;
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) return 0;
//   TIter next(fPhysical);
   // Ignore the request if the current state in the main geometry matches the
   // last touched physical node in the parallel geometry
   if (fLastState && fLastState->IsMatchingState(nav)) return 0;
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
   if (nd<5) {
   // loop over daughters
      for (i=0; i<nd; i++) {
         current = fVolume->GetNode(i);
         // validate only within stepmax
         if (voxels->IsSafeVoxel(point, i, stepmax)) continue;
         current->MasterToLocal(point, lpoint);
         current->MasterToLocalVect(dir, ldir);
         Double_t snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, step);
         if (snext < step) {
            step = snext;
            idaughter = i;
         }
      }
      if (idaughter>=0) {
         pnode = (TGeoPhysicalNode*)fPhysical->At(idaughter);
         return pnode;
      }
      step = TGeoShape::Big();
      return 0;
   }
   // Get current voxel
   Int_t ncheck = 0;
   Int_t sumchecked = 0;
   Int_t *vlist = 0;
   TGeoNodeCache *cache = nav->GetCache();
   TGeoStateInfo &info = *cache->GetMakePWInfo(nd);
//   TGeoStateInfo &info = *cache->GetInfo();
//   cache->ReleaseInfo(); // no hierarchical use
   voxels->SortCrossedVoxels(point, dir, info);
   while ((sumchecked<nd) && (vlist=voxels->GetNextVoxel(point, dir, ncheck, info))) {
      for (i=0; i<ncheck; i++) {
         pnode = (TGeoPhysicalNode*)fPhysical->At(vlist[i]);
         if (pnode->IsMatchingState(nav)) {
            step = TGeoShape::Big();
            return 0;
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
      if (idaughter>=0) {
         pnode = (TGeoPhysicalNode*)fPhysical->At(idaughter);
         // mark the overlap
         if (!fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) {
            AddOverlap(nav->GetCurrentVolume(),kFALSE);
//            printf("object %s overlapping with pn: %s\n", fGeoManager->GetPath(), pnode->GetName());
         }
         return pnode;
      }
   }
   step = TGeoShape::Big();
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute safety for the parallel world

Double_t TGeoParallelWorld::Safety(Double_t point[3], Double_t safmax)
{
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if the state matches the last one recorded
   if (fLastState && fLastState->IsMatchingState(nav)) return TGeoShape::Big();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) return TGeoShape::Big();
   Double_t local[3];
   Double_t safe = safmax;
   Double_t safnext;
   TGeoPhysicalNode *pnode = 0;
   const Double_t tolerance = TGeoShape::Tolerance();
   Int_t nd = fVolume->GetNdaughters();
   TGeoNode *current;
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   //---> check fast unsafe voxels
   Double_t *boxes = voxels->GetBoxes();
   for (Int_t id=0; id<nd; id++) {
      Int_t ist = 6*id;
      Double_t dxyz = 0.;
      Double_t dxyz0 = TMath::Abs(point[0]-boxes[ist+3])-boxes[ist];
      if (dxyz0 > safe) continue;
      Double_t dxyz1 = TMath::Abs(point[1]-boxes[ist+4])-boxes[ist+1];
      if (dxyz1 > safe) continue;
      Double_t dxyz2 = TMath::Abs(point[2]-boxes[ist+5])-boxes[ist+2];
      if (dxyz2 > safe) continue;
      if (dxyz0>0) dxyz+=dxyz0*dxyz0;
      if (dxyz1>0) dxyz+=dxyz1*dxyz1;
      if (dxyz2>0) dxyz+=dxyz2*dxyz2;
      if (dxyz >= safe*safe) continue;
      pnode = (TGeoPhysicalNode*)fPhysical->At(id);
      // Return if inside the current node
      if (pnode->IsMatchingState(nav)) return TGeoShape::Big();
      current = fVolume->GetNode(id);
      current->MasterToLocal(point, local);
      // Safety to current node
      safnext = current->Safety(local, kFALSE);
      if (safnext < tolerance) return 0.;
      if (safnext < safe) safe = safnext;
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

