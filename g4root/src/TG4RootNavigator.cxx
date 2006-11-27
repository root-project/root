// @(#)root/g4root:$Name:  $:$Id: TG4RootNavigator.cxx,v 1.2 2006/11/22 17:29:54 rdm Exp $
// Author: Andrei Gheata   07/08/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TG4Navigator                                                         //
//                                                                      //
// GEANT4 navigator using directly a TGeo geometry.                     //
//                                                                      //
// All navigation methods requred by G4 tracking are implemented by     //
// this class by invoking the corresponding functionality of ROOT       //
// geometry modeler.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoManager.h"

#include "TG4RootDetectorConstruction.h"
#include "TG4RootNavigator.h"

//ClassImp(TG4RootNavigator)
static const double gCm = 1./cm;

//______________________________________________________________________________
TG4RootNavigator::TG4RootNavigator()
                 :G4Navigator(),
                  fGeometry(0),
                  fDetConstruction(0),
                  fStepEntering(kFALSE),
                  fStepExiting(kFALSE),
                  fNextPoint()
{
// Dummy ctor.
}

//______________________________________________________________________________
TG4RootNavigator::TG4RootNavigator(TG4RootDetectorConstruction *dc)
                 :G4Navigator(),
                  fGeometry(0),
                  fDetConstruction(0),
                  fStepEntering(kFALSE),
                  fStepExiting(kFALSE),
                  fNextPoint()
{
// Default ctor.
   SetDetectorConstruction(dc);
   SetWorldVolume(dc->GetTopPV());
}

//______________________________________________________________________________
TG4RootNavigator::~TG4RootNavigator()
{
// Destructor.
}

//______________________________________________________________________________
void TG4RootNavigator::SetDetectorConstruction(TG4RootDetectorConstruction *dc)
{
// Setter for detector construction. Root geometry manager pointer is taken from
// it and must be valid.
   if (dc) fGeometry = dc->GetGeometryManager();
   if (!fGeometry || !fGeometry->IsClosed()) {
      G4cerr << "Cannot create TG4RootNavigator without closed ROOT geometry !" << G4endl;
      G4Exception("Aborting...");
   }   
   fDetConstruction = dc;
}
  
//______________________________________________________________________________
G4double TG4RootNavigator::ComputeStep(const G4ThreeVector &pGlobalPoint,
                                       const G4ThreeVector &pDirection,
                                       const G4double pCurrentProposedStepLength,
                                       G4double  &pNewSafety)
{
// Calculate the distance to the next boundary intersected
// along the specified NORMALISED vector direction and
// from the specified point in the global coordinate
// system. LocateGlobalPointAndSetup or LocateGlobalPointWithinVolume 
// must have been called with the same global point prior to this call.
// The isotropic distance to the nearest boundary is also
// calculated (usually an underestimate). The current
// proposed Step length is used to avoid intersection
// calculations: if it can be determined that the nearest
// boundary is >pCurrentProposedStepLength away, kInfinity
// is returned together with the computed isotropic safety
// distance. Geometry must be closed.


   // The following 2 lines are not needed if G4 calls first LocateGlobalPoint...
//   fGeometry->ResetState();
   static Long64_t istep = 0;
   istep++;
//   printf("step#%lld\n", istep);
   Double_t tol = 0.;
   if (fEnteredDaughter || fExitedMother) {
      Double_t npt[3];
      tol = TGeoShape::Tolerance();
      npt[0] = pGlobalPoint.x()*gCm+tol*pDirection.x();
      npt[1] = pGlobalPoint.y()*gCm+tol*pDirection.y();
      npt[2] = pGlobalPoint.z()*gCm+tol*pDirection.z();
      fGeometry->SetCurrentPoint(npt[0],npt[1],npt[2]);
   } else {   
      fGeometry->SetCurrentPoint(pGlobalPoint.x()*gCm, pGlobalPoint.y()*gCm, pGlobalPoint.z()*gCm);
   }   
   fGeometry->SetCurrentDirection(pDirection.x(), pDirection.y(), pDirection.z());
   fGeometry->FindNextBoundary(-(pCurrentProposedStepLength*gCm-tol));
   pNewSafety = (fGeometry->GetSafeDistance()-tol)*cm;
   if (pNewSafety<0.) pNewSafety = 0.;
   G4double step = (gGeoManager->GetStep()+tol)*cm;
//   if (step >= pCurrentProposedStepLength) step = kInfinity;
   if (step < 2.*tol*cm) step = 0.;
   fStepEntering = fGeometry->IsStepEntering();
   fStepExiting  = fGeometry->IsStepExiting();
   if (fStepEntering || fStepExiting) {
      fNextPoint = pGlobalPoint + step*pDirection;
   } else {
      step = kInfinity;
   }  
//   G4cout.precision(12);
//   G4cout << "ComputeStep: point=" << pGlobalPoint << " dir=" << pDirection << G4endl;
//   G4cout << "             pstep="<<pCurrentProposedStepLength << " snext=" << step << G4endl;
//   G4cout << "             safe ="<<pNewSafety<< "  onBound="<<fGeometry->IsOnBoundary()<<" entering=" <<fStepEntering << " exiting="<<fStepExiting << G4endl;
   return step;
}   

//______________________________________________________________________________
G4VPhysicalVolume* TG4RootNavigator::ResetHierarchyAndLocate(
                                       const G4ThreeVector &point,
                                       const G4ThreeVector &direction,
                                       const G4TouchableHistory &h)
{
// Resets the geometrical hierarchy and search for the volumes deepest
// in the hierarchy containing the point in the global coordinate space.
// The direction is used to check if a volume is entered.
// The search begin is the geometrical hierarchy at the location of the
// last located point, or the endpoint of the previous Step if
// SetGeometricallyLimitedStep() has been called immediately before.
// 
// Important Note: In order to call this the geometry MUST be closed.
//
// In case of TGeo-based geometry all volumes look as normal positioned, so 
// there is no need to reset the hierarchy. The state of TGeo needs however
// to be synchronized.
//   G4cout << "ResetHierarchyAndLocate: POINT: " << point << " DIR: "<< direction << G4endl;
//   G4cout << "ResetHierarchyAndLocate: point=" << point << G4endl;
   ResetState();
   fHistory = *h.GetHistory();
   SynchronizeGeoManager();
   fGeometry->InitTrack(point.x()*gCm, point.y()*gCm, point.z()*gCm, direction.x(), direction.y(), direction.z());
   G4VPhysicalVolume *pVol = SynchronizeHistory();
//   G4cout << fHistory << G4endl;
   return pVol;
}
   
//______________________________________________________________________________
TGeoNode *TG4RootNavigator::SynchronizeGeoManager()
{
// Synchronize the current state of TGeoManager with the current navigation
// history. Do the minimum possible work in case 
// states are already (or almost) in sync. Returns current logical node.
   Int_t geolevel = fGeometry->GetLevel();
   Int_t depth = fHistory.GetDepth();
   Int_t nodeIndex, level;
   G4VPhysicalVolume *pvol;
   TGeoNode *pnode, *newnode=0;
   for (level=0; level<=depth; level++) {
      pvol = fHistory.GetVolume(level);
      if (level<=geolevel) {
         // TGeo has also something at this level - check if it matches what is
         // in fHistory
         pnode = fGeometry->GetMother(geolevel-level);
         newnode = fDetConstruction->GetNode(pvol);
         // If the node at this level matches the one in the history, do nothing
         if (pnode==newnode) continue;
         // From this level down we need to update TGeo path.
         if (level==0) {
            // TO BE REMOVED IF NEVER HAPPENS !!!
            G4cerr << "Top node does not match history !!!" << G4endl;
            G4Exception("Aborting in SynchronizeGeoManager()");
            return NULL;
         }
         while (geolevel >= level) {
            fGeometry->CdUp();
            geolevel--;
         }
         // Now TGeo is at level-1 and needs to update level
         // this should be the index of the node to be used in CdDown(index)
         nodeIndex = fHistory.GetReplicaNo(level);
         fGeometry->CdDown(nodeIndex);
         geolevel++;     // Should be equal to i now
      } else {
         // This level has to be synchronized
         nodeIndex = fHistory.GetReplicaNo(level);
         fGeometry->CdDown(nodeIndex);
         // Do the check for the moment - TO REMOVE AFTER CHECK !!!!!
         if (fGeometry->GetCurrentNode() != fDetConstruction->GetNode(fHistory.GetVolume(level))) {
            G4cerr << "WOOPS: CdDown(fHistory.GetReplica) did not work !!!" << G4endl;
            return NULL;
         }   
         geolevel++;     // Should be equal to i now
      }
   }
   return fGeometry->GetCurrentNode();
}          
      
//______________________________________________________________________________
G4VPhysicalVolume *TG4RootNavigator::SynchronizeHistory()
{
// Synchronize the current navigation history according the state of TGeoManager
// Do the minimum possible work in case states are already (or almost) in sync.
// Returns current physical volume
   Int_t depth = fHistory.GetDepth();
   Int_t geolevel = fGeometry->GetLevel();
   G4VPhysicalVolume *pvol, *pnewvol=0;
   TGeoNode *pnode;
   Int_t level;
   for (level=0; level<=geolevel; level++) {
      pnode = fGeometry->GetMother(geolevel-level);
      pnewvol = fDetConstruction->GetG4VPhysicalVolume(pnode);
      if (level<=depth) {
         pvol = fHistory.GetVolume(level);
         // If the phys. volume at this level matches the one in the history, do nothing
         if (pvol==pnewvol) continue;
         // From this level down we need to update G4 history.
         if (level) {
            fHistory.BackLevel(depth-level+1);
            // Now fHistory is at the level i-1 and needs to update level i
            fHistory.NewLevel(pnewvol, kNormal, pnewvol->GetCopyNo());
         } else {
            // We need to refresh top level
            fHistory.BackLevel(depth);
            fHistory.SetFirstEntry(pnewvol);
         }
         depth = level;
      } else {
         // This level has to be added to the current history.
         fHistory.NewLevel(pnewvol, kNormal, pnewvol->GetCopyNo());
         depth++;     // depth=level
      }
   }
   if (depth > level-1) fHistory.BackLevel(depth-level+1);
   if (fGeometry->IsOutside()) pnewvol = NULL;      
   return pnewvol;      
}         

//______________________________________________________________________________
G4VPhysicalVolume* 
TG4RootNavigator::LocateGlobalPointAndSetup(const G4ThreeVector& globalPoint,
                                            const G4ThreeVector* pGlobalDirection,
                                            const G4bool relativeSearch,
                                            const G4bool ignoreDirection)
{
// Locate the point in the hierarchy return 0 if outside
// The direction is required 
//    - if on an edge shared by more than two surfaces 
//      (to resolve likely looping in tracking)
//    - at initial location of a particle
//      (to resolve potential ambiguity at boundary)
// 
// Flags on exit: (comments to be completed)
// fEntering         - True if entering `daughter' volume (or replica)
//                     whether daughter of last mother directly 
//                     or daughter of that volume's ancestor.

   static Long64_t ilocate = 0;
   ilocate++;
//   printf("locate#%lld\n", ilocate);
//   G4cout.precision(12);
//   G4cout << "LocateGlobalPointAndSetup: point: " << globalPoint << G4endl;
   fGeometry->SetCurrentPoint(globalPoint.x()*gCm, globalPoint.y()*gCm, globalPoint.z()*gCm);
   fEnteredDaughter = fExitedMother = kFALSE;
   Bool_t onBoundary = kFALSE;
   if (fStepEntering || fStepExiting) {
      Double_t d2 = (globalPoint.x()-fNextPoint.x())*(globalPoint.x()-fNextPoint.x()) +
                    (globalPoint.y()-fNextPoint.y())*(globalPoint.y()-fNextPoint.y()) +
                    (globalPoint.z()-fNextPoint.z())*(globalPoint.z()-fNextPoint.z());
      if (d2 < 1.e-20) onBoundary = kTRUE;
//      G4cout << G4endl << "    ON BOUNDARY" << "entering/exiting="<< fStepEntering << "/" << fStepExiting << G4endl;
   }
   if ((!ignoreDirection || onBoundary )&& pGlobalDirection) {
      fGeometry->SetCurrentDirection(pGlobalDirection->x(), pGlobalDirection->y(), pGlobalDirection->z());
   }
//   G4cout << "    init History = " << G4endl << fHistory << G4endl;
//   printf("level %i: %s\n", fGeometry->GetLevel(), fGeometry->GetPath());
//   if (fGeometry->IsOutside()) G4cout << "   outside" << G4endl;
   if (onBoundary) {
      fEnteredDaughter = fStepEntering;
      fExitedMother    = fStepExiting;
      TGeoNode *skip = fGeometry->GetCurrentNode();
      if (fStepExiting) {
         if (!fGeometry->GetLevel()) {
            fGeometry->SetOutside();
            return NULL;
         }   
         fGeometry->CdUp();
      } else {
         if (fStepEntering && fGeometry->IsOutside()) skip = 0;
      }      
      fGeometry->CrossBoundaryAndLocate(fStepEntering, skip);
//      if (fGeometry->IsSameLocation()) fForceCross = kTRUE;
   } else {   
      if (!relativeSearch) fGeometry->CdTop();
      fGeometry->FindNode();
   }   
   G4VPhysicalVolume *target = SynchronizeHistory();
//   if (fGeometry->IsSameLocation()) {
//      fEnteredDaughter = fExitedMother = kFALSE;
//   } else {
//      fEnteredDaughter = fExitedMother = kTRUE;   
//   }   
//   G4cout << "    out History = " << G4endl << fHistory << G4endl;
//   if (fGeometry->IsOutside()) G4cout << "   outside" << G4endl;
   return target;
}
   
//______________________________________________________________________________
void TG4RootNavigator::LocateGlobalPointWithinVolume(const G4ThreeVector& pGlobalPoint)
{
// Notify the Navigator that a track has moved to the new Global point
// 'position', that is known to be within the current safety.
// No check is performed to ensure that it is within  the volume. 
// This method can be called instead of LocateGlobalPointAndSetup ONLY if
// the caller is certain that the new global point (position) is inside the
// same volume as the previous position.  Usually this can be guaranteed
// only if the point is within safety.
//   fLastLocatedPointLocal = ComputeLocalPoint(pGlobalPoint);
//   G4cout << "LocateGlobalPointWithinVolume: POINT: " << pGlobalPoint << G4endl;
//   printf("LocateGlobalPointWithinVolume: point=(%g,%g,%g)\n", pGlobalPoint.x(),pGlobalPoint.y(),pGlobalPoint.z());
   fGeometry->SetCurrentPoint(pGlobalPoint.x()*gCm, pGlobalPoint.y()*gCm, pGlobalPoint.z()*gCm);
   fStepEntering = kFALSE;
   fStepExiting = kFALSE;
}

//______________________________________________________________________________
/*
void TG4RootNavigator::LocateGlobalPointAndUpdateTouchableHandle(
                                       const G4ThreeVector &position,
                                       const G4ThreeVector &direction,
                                       G4TouchableHandle   &oldTouchableToUpdate,
                                       const G4bool        relativeSearch)
{
// First, search the geometrical hierarchy like the above method
// LocateGlobalPointAndSetup(). Then use the volume found and its
// navigation history to update the touchable.
//   G4cout << "LocateGlobalPointAndUpdateTouchableHandle: POINT: " << position << G4endl;
   G4VPhysicalVolume* pPhysVol;
   pPhysVol = LocateGlobalPointAndSetup( position,&direction,relativeSearch );
   if(!fGeometry->IsSameLocation()) {
      oldTouchableToUpdate = CreateTouchableHistory();
      if( pPhysVol == 0 ) {
         // We want to ensure that the touchable is correct in this case.
         //  The method below should do this and recalculate a lot more ....
         //
         oldTouchableToUpdate->UpdateYourself( pPhysVol, &fHistory );
      }
   }
}
*/
//______________________________________________________________________________
G4double TG4RootNavigator::ComputeSafety(const G4ThreeVector &globalpoint, 
                                         const G4double /*pProposedMaxLength*/)
{
// Calculate the isotropic distance to the nearest boundary from the
// specified point in the global coordinate system. 
// The globalpoint utilised must be within the current volume.
// The value returned is usually an underestimate.  
// The proposed maximum length is used to avoid volume safety
// calculations.  The geometry must be closed.

// TO CHANGE TGeoManager::Safety To take into account pProposedMaxLength
   fGeometry->ResetState();
   fGeometry->SetCurrentPoint(globalpoint.x()*gCm, globalpoint.y()*gCm, globalpoint.z()*gCm);
   G4double safety = fGeometry->Safety()*cm;
//   G4cout << "ComputeSafety: POINT: " << globalpoint << " safe = " << safety << G4endl;
   return safety;
}
   
//______________________________________________________________________________
G4TouchableHistoryHandle TG4RootNavigator::CreateTouchableHistoryHandle() const
{
// Returns a reference counted handle to a touchable history.
   return G4Navigator::CreateTouchableHistoryHandle();
}

//______________________________________________________________________________
G4ThreeVector TG4RootNavigator::GetLocalExitNormal(G4bool* valid)
{
// Returns Exit Surface Normal and validity too.
// It can only be called if the Navigator's last Step has crossed a
// volume geometrical boundary.
// It returns the Normal to the surface pointing out of the volume that
// was left behind and/or into the volume that was entered.
// (The normal is in the coordinate system of the final volume.)
// This function takes full care about how to calculate this normal,
// but if the surfaces are not convex it will return valid=false.
   Double_t *norm, lnorm[3];
   *valid = true;
   norm = fGeometry->FindNormalFast();
   G4ThreeVector normal(0.,0.,1.);
   if (!norm) {
      *valid = false;
      return normal;
   }
   fGeometry->MasterToLocalVect(norm, lnorm);
   normal.setX(lnorm[0]);   
   normal.setY(lnorm[1]);   
   normal.setZ(lnorm[2]);  
//   G4cout << "GetLocalExitNormal: " << normal << G4endl;
//   G4cout << "GetLocalExitNormal: " << normal << G4endl;   
   return normal; 
}
