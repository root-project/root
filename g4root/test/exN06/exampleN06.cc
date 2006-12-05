//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// $Id: exampleN06.cc,v 1.14 2006/06/29 17:53:52 gunter Exp $
// GEANT4 tag $Name: geant4-08-01 $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//
// Description: Test of Continuous Process G4Cerenkov
//              and RestDiscrete Process G4Scintillation
//              -- Generation Cerenkov Photons --
//              -- Generation Scintillation Photons --
//              -- Transport of optical Photons --
// Version:     5.0
// Created:     1996-04-30
// Author:      Juliet Armstrong
// mail:        gum@triumf.ca
//     
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4UIterminal.hh"
#include "G4UItcsh.hh"

#include "G4ios.hh"

#include "ExN06DetectorConstruction.hh"
#include "ExN06PhysicsList.hh"
#include "ExN06PrimaryGeneratorAction.hh"
#include "ExN06RunAction.hh"
#include "ExN06EventAction.hh"
#include "ExN06StackingAction.hh"
#include "ExN06SteppingVerbose.hh"
#include "ExN06PostDetConstruction.hh"

#include "TGeoManager.h"
#include "TG4RootDetectorConstruction.h"
#include "TG4RootNavMgr.h"

#include "Randomize.hh"

#ifdef G4VIS_USE
#include "G4VisExecutive.hh"
#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

int main(int argc,char** /*argv*/)
{
  // Seed the random number generator manually
  //
  G4long myseed = 345354;
  
  // User Verbose output class
  //
  G4VSteppingVerbose* verbosity = new ExN06SteppingVerbose;
  G4VSteppingVerbose::SetInstance(verbosity);
  
  G4bool use_tgeo = (argc>1)?kTRUE:kFALSE;
  TGeoManager *geom = 0;
  TG4RootNavMgr *mgr = 0;
  // Run manager
  //
  if (use_tgeo) {
     G4cout << "Using TGeo interface ..." << G4endl;
     geom = TGeoManager::Import("ex06geom.root");
     mgr = TG4RootNavMgr::GetInstance(geom);
  } else {
     G4cout << "Using G4 native ..." << G4endl;  
  }
  G4RunManager* runManager = new G4RunManager;
  if (use_tgeo) {
     mgr->Initialize(ExN06PostDetConstruction::GetInstance());
     mgr->ConnectToG4();
  }   

  // UserInitialization classes - mandatory;
  //
  G4VUserDetectorConstruction* detector = new ExN06DetectorConstruction;
  if (!use_tgeo) runManager-> SetUserInitialization(detector);
  //
  G4VUserPhysicsList* physics = new ExN06PhysicsList;
  runManager-> SetUserInitialization(physics);

  // UserAction classes
  //
  G4UserRunAction* run_action = new ExN06RunAction;
  runManager->SetUserAction(run_action);
  //
  G4VUserPrimaryGeneratorAction* gen_action = new ExN06PrimaryGeneratorAction;
  runManager->SetUserAction(gen_action);
  //
  G4UserEventAction* event_action = new ExN06EventAction;
  runManager->SetUserAction(event_action);
  //
  G4UserStackingAction* stacking_action = new ExN06StackingAction;
  runManager->SetUserAction(stacking_action);
  
  // Initialize G4 kernel
  //
  runManager->Initialize();
  CLHEP::HepRandom::setTheSeed(myseed);

  G4cout << *(G4Material::GetMaterialTable()) << G4endl;
  // Get the pointer to the User Interface manager
  //
  G4UImanager* UI = G4UImanager::GetUIpointer();    
  UI->ApplyCommand("/control/execute exN06.in"); 
  if (use_tgeo) 
     ExN06PostDetConstruction::GetInstance()->WriteTracks("tracks_tgeo.root");
  else
     ExN06PostDetConstruction::GetInstance()->WriteTracks("tracks_g4.root");

  delete runManager;
  delete verbosity;

  return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
