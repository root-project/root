// Author: Andrei Gheata   17/02/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_____________________________________________________________________________
// TGeoParallelWorld    - base class for a flat parallel geometry.
//   The parallel geometry can be composed by both normal volumes added
// using the AddNode interface (not implemented yet) or by physical nodes 
// which will use as position their actual global matrix with respect to the top 
// volume of the main geometry. 
//   All these nodes are added as daughters to the "top" volume of
// the parallel world which acts as a navigation helper in this parallel
// world. The parallel world has to be closed before calling any navigation
// method.
//_____________________________________________________________________________

#include "TGeoParallelWorld.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoVoxelFinder.h"
#include "TGeoMatrix.h"
#include "TGeoPhysicalNode.h"
#include "TGeoNavigator.h"

ClassImp(TGeoParallelWorld)

//_____________________________________________________________________________
TGeoParallelWorld::TGeoParallelWorld(const char *name, TGeoManager *mgr) 
                  : TNamed(name,""),
                    fGeoManager(mgr),
                    fPhysical(0),
                    fVolume(new TGeoVolumeAssembly(name)),
                    fIsClosed(kFALSE),
                    fUseOverlaps(kFALSE)
{
// Default constructor
}

//_____________________________________________________________________________
TGeoParallelWorld::~TGeoParallelWorld()
{
// Destructor
   delete fPhysical;
}

//_____________________________________________________________________________
void TGeoParallelWorld::AddNode(TGeoPhysicalNode *pnode)
{
// Add a node normally to this world. Overlapping nodes not allowed
   if (fIsClosed) Fatal("AddNode", "Cannot add nodes to a closed parallel geometry");
   if (!fPhysical) fPhysical = new TObjArray(256);
   fPhysical->Add(pnode);
}

//_____________________________________________________________________________
void TGeoParallelWorld::AddOverlap(TGeoVolume *vol)
{
// To use this optimization, the user should declare the full list of volumes
// which may overlap with any of the physical nodes of the parallel world. Better
// be done before misalignment
   fUseOverlaps = kTRUE;
   vol->SetOverlappingCandidate(kTRUE);
}

//_____________________________________________________________________________
Bool_t TGeoParallelWorld::CloseGeometry()
{
// The main geometry must be closed.
   if (fIsClosed) return kTRUE;
   if (!fGeoManager->IsClosed()) Fatal("CloseGeometry", "Main geometry must be closed first");
   if (!fPhysical || !fPhysical->GetEntriesFast()) {
      Error("CloseGeometry", "List of physical nodes is empty");
      return kFALSE;
   }
   RefreshPhysicalNodes();
   fIsClosed = kTRUE;
   return kTRUE;
}   

//_____________________________________________________________________________
void TGeoParallelWorld::RefreshPhysicalNodes()
{
// Refresh the node pointers and re-voxelize. To be called mandatory in case 
// re-alignment happened.
   if (fIsClosed) {
      delete fVolume;
      fVolume = new TGeoVolume();
   }
   // Loop physical nodes and add them to the navigation helper volume
   TGeoPhysicalNode *pnode;
   TIter next(fPhysical);
   Int_t copy = 0;
   while ((pnode = (TGeoPhysicalNode*)next())) {
      fVolume->AddNode(pnode->GetVolume(), copy++, new TGeoHMatrix(*pnode->GetMatrix()));
   }
   // Voxelize the volume
   fVolume->GetShape()->ComputeBBox();
   fVolume->Voxelize("ALL");
}   

//_____________________________________________________________________________
TGeoPhysicalNode *TGeoParallelWorld::FindNode(Double_t point[3])
{
// Finds physical node containing the point
   if (!fIsClosed) Fatal("FindNode", "Parallel geometry must be closed first");
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) return 0;
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   Int_t id;
   Int_t ncheck = 0;
   // get the list of nodes passing thorough the current voxel
   TGeoNodeCache *cache = nav->GetCache();
   TGeoStateInfo &info = *cache->GetInfo();
   Int_t *check_list = voxels->GetCheckList(point, ncheck, info);
   cache->ReleaseInfo(); // no hierarchical use
   if (!check_list) return 0;
   // loop all nodes in voxel
   TGeoNode *node;
   TGeoPhysicalNode *pnode;
   Double_t local[3];
   for (id=0; id<ncheck; id++) {
      node = fVolume->GetNode(check_list[id]);
      node->MasterToLocal(point, local);
      if (node->GetVolume()->Contains(local)) {
         // We found a node containing the point
         pnode = (TGeoPhysicalNode*)fPhysical->At(node->GetNumber());
         return pnode;
      }
   }
   return 0;
}   

//_____________________________________________________________________________
TGeoPhysicalNode *TGeoParallelWorld::FindNextBoundary(Double_t point[3], Double_t dir[3],
                              Double_t &step, Double_t stepmax)
{
// Same functionality as TGeoNavigator::FindNextDaughterBoundary for the
// parallel world
   if (!fIsClosed) Fatal("FindNode", "Parallel geometry must be closed first");
   TGeoPhysicalNode *pnode = 0;
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) return 0;
   TIter next(fPhysical);
   // Ignore the request if the current state in the main geometry matches one
   // of the physical nodes in the parallel geometry
   while ((pnode = (TGeoPhysicalNode*)next())) {
      if (pnode->IsMatchingState(nav)) return 0;
   }   
   Double_t snext = TGeoShape::Big();
   step = stepmax;
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   Int_t idaughter = -1; // nothing crossed
   Int_t nd = fVolume->GetNdaughters();
   Int_t i;
   TGeoNode *current;
   Double_t lpoint[3], ldir[3];
   const Double_t tolerance = TGeoShape::Tolerance();
   if (nd<5) {
   // loop over daughters
      for (i=0; i<nd; i++) {
         current = fVolume->GetNode(i);
         // validate only within stepmax
         if (voxels->IsSafeVoxel(point, i, stepmax)) continue;
         current->MasterToLocal(point, lpoint);
         current->MasterToLocalVect(dir, ldir);
         snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, step);
         if (snext < step-tolerance) {
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
   TGeoStateInfo &info = *cache->GetInfo();
   cache->ReleaseInfo(); // no hierarchical use
   voxels->SortCrossedVoxels(point, dir, info);
   while ((sumchecked<nd) && (vlist=voxels->GetNextVoxel(point, dir, ncheck, info))) {
      for (i=0; i<ncheck; i++) {
         current = fVolume->GetNode(vlist[i]);
         current->MasterToLocal(point, lpoint);
         current->MasterToLocalVect(dir, ldir);
         snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, step);
         if (snext < step-tolerance) {
            step = snext;
            idaughter = vlist[i];
         }
      }   
      if (idaughter>=0) {
         pnode = (TGeoPhysicalNode*)fPhysical->At(idaughter);
         return pnode;
      }
   }   
   step = TGeoShape::Big();
   return 0;
}   

//_____________________________________________________________________________
Double_t TGeoParallelWorld::Safety(Double_t point[3], Double_t safmax)
{
// Compute safety for the parallel world
   TGeoNavigator *nav = fGeoManager->GetCurrentNavigator();
   // Fast return if not in an overlapping candidate
   if (fUseOverlaps && !nav->GetCurrentVolume()->IsOverlappingCandidate()) return TGeoShape::Big();
   Double_t local[3];
   Double_t safe = safmax;
   Double_t safnext;
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
      current = fVolume->GetNode(id);
      current->MasterToLocal(point, local);
      // Safety to current node
      safnext = current->Safety(local, kFALSE);
      if (safnext < tolerance) return 0.;
      if (safnext < safe) safe = safnext;
   }
   return safe;
}   

//_____________________________________________________________________________
void TGeoParallelWorld::CheckOverlaps(Double_t ovlp)
{
// Check overlaps within a tolerance value.
   fVolume->CheckOverlaps(ovlp);
}
   
//_____________________________________________________________________________
void TGeoParallelWorld::Draw(Option_t *option)
{
// Draw the parallel world
   fVolume->Draw(option);
}
   
