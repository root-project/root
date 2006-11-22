// @(#):$Name:  $:$Id: Exp $
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
// TG4RootNavMgr                                                        //
//                                                                      //
// Manager class creating a G4Navigator based on a ROOT geometry.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGeoManager.h"
#include "TG4RootNavigator.h"
#include "TG4RootDetectorConstruction.h"
#include "TG4RootNavMgr.h"

#include "G4RunManager.hh"
#include "G4TransportationManager.hh"
#include "G4PropagatorInField.hh"

ClassImp(TG4RootNavMgr)

TG4RootNavMgr *TG4RootNavMgr::fRootNavMgr = 0;

//______________________________________________________________________________
TG4RootNavMgr::TG4RootNavMgr()
              :TObject(),
               fGeometry(0),
               fNavigator(0),
               fDetConstruction(0),
               fConnected(kFALSE)
{
// Dummy ctor.
}

//______________________________________________________________________________
TG4RootNavMgr::TG4RootNavMgr(TGeoManager *geom)
              :TObject(),
               fGeometry(0),
               fNavigator(0),
               fDetConstruction(0),
               fConnected(kFALSE)
{
// Default ctor.
   fDetConstruction = new TG4RootDetectorConstruction(geom);
   SetNavigator(new TG4RootNavigator);
}

//______________________________________________________________________________
TG4RootNavMgr::~TG4RootNavMgr()
{
// Destructor.
//   if (fNavigator) delete fNavigator;
   if (fDetConstruction) delete fDetConstruction;
   fRootNavMgr = 0;
}

//______________________________________________________________________________
TG4RootNavMgr *TG4RootNavMgr::GetInstance(TGeoManager *geom)
{
// Get the pointer to the singleton. If none, create one based on 'geom'.
   if (fRootNavMgr) return fRootNavMgr;
   // Check if we have to create one.
   if (!geom) return NULL;
   fRootNavMgr = new TG4RootNavMgr(geom);
   return fRootNavMgr;
}

//______________________________________________________________________________
Bool_t TG4RootNavMgr::ConnectToG4()
{
// Connect detector construction class to G4 run manager.
   if (fConnected) {
      Info("ConnectToG4", "Already connected");
      return kTRUE;
   }   
   if (!fDetConstruction) {
      Error("ConnectToG4", "No detector construction set !");
      return kFALSE;
   }   
   
   if (!fNavigator) {
      Error("ConnectToG4", "Navigator has to be created befor connecting to G4 !!!");
      return kFALSE;
   }   

   G4RunManager *runManager = G4RunManager::GetRunManager();
   if (!runManager) {
      Error("ConnectToG4", "Unable to connect: G4RunManager not instantiated");
      return kFALSE;
   }
   runManager->SetUserInitialization(fDetConstruction);
   Info("ConnectToG4", "ROOT detector construction class connected to G4RunManager");
   fConnected = kTRUE;
   return kTRUE;
}   

//______________________________________________________________________________
void TG4RootNavMgr::SetNavigator(TG4RootNavigator *nav)
{
// Connect a navigator to G4.
   if (fConnected) {
      Error("SetNavigator", "Navigator set after instantiation of G4RunManager. Won't set!!!");
      return;
   }   
   G4TransportationManager *trMgr = G4TransportationManager::GetTransportationManager();
//   G4Navigator *oldNav = trMgr->GetNavigatorForTracking();
   trMgr->SetNavigatorForTracking(nav);
   G4FieldManager *fieldMgr = trMgr->GetPropagatorInField()->GetCurrentFieldManager();
   delete trMgr->GetPropagatorInField();
   trMgr->SetPropagatorInField(new G4PropagatorInField(nav, fieldMgr));
   trMgr->ActivateNavigator(nav);
   fNavigator = nav;
//   trMgr->DeRegisterNavigator(oldNav);
   Info("SetNavigator", "TG4RootNavigator created and registered to G4TransportationManager");
}

//______________________________________________________________________________
void TG4RootNavMgr::Initialize(TVirtualUserPostDetConstruction *sdinit)
{
// Construct G4 geometry based on TGeo geometry. 
   Info("Initialize", "Creating G4 hierarchy ...");
   if (fDetConstruction) fDetConstruction->Initialize(sdinit);
}

//______________________________________________________________________________
void TG4RootNavMgr::LocateGlobalPointAndSetup(Double_t *pt, Double_t *dir)
{
// Test the corresponding navigation method.
   G4ThreeVector point(pt[0], pt[1], pt[2]);
   G4VPhysicalVolume *pVol = 0;
   if (dir) {
      G4ThreeVector direction(dir[0], dir[1], dir[2]);
      pVol = fNavigator->LocateGlobalPointAndSetup(point, &direction);
   } else {
      pVol = fNavigator->LocateGlobalPointAndSetup(point);
   }
   fNavigator->PrintState();
}   

//______________________________________________________________________________
void TG4RootNavMgr::SetVerboseLevel(Int_t level)
{
// Set navigator verbosity level.
   fNavigator->SetVerboseLevel(level);
}

//______________________________________________________________________________
void TG4RootNavMgr::PrintG4State() const
{
// Print current G4 state.
   G4NavigationHistory *history = fNavigator->GetHistory();
   G4cout << *history << G4endl;
}
