//
// Macro for loading Geant4 and Geant4 VMC libraries


#include <iostream>
void g4libs()
{
   g4libs_granular();
}
   
void g4libs_granular()
{
// Loads G4 granular libraries and G4root library. 
// external packages: CLHEP, used by G4
// ---

  cout << "Loading Geant4 granular libraries ..." << endl;

  // CLHEP
  gSystem->Load("libCLHEP");

  // G4 categories

  // global
  gSystem->Load("libG4globman");  
  gSystem->Load("libG4hepnumerics");
  // graphics_reps
  gSystem->Load("libG4graphics_reps");   
  // intercoms
  gSystem->Load("libG4intercoms");
  // materials
  gSystem->Load("libG4materials");
  // geometry
  gSystem->Load("libG4geometrymng");  
  gSystem->Load("libG4magneticfield");
  gSystem->Load("libG4volumes");
  gSystem->Load("libG4navigation");
  gSystem->Load("libG4geombias");
  // particles  
  gSystem->Load("libG4partman");
  // track
  gSystem->Load("libG4track");
  // tracking
  gSystem->Load("libG4tracking");
  // digits_hits  
  gSystem->Load("libG4hits");
  gSystem->Load("libG4digits");   
  gSystem->Load("libG4detector");   
  // event
  gSystem->Load("libG4event");  
  // readout
  gSystem->Load("libG4readout");
  // run
  gSystem->Load("libG4run");
  // geom
  gSystem->Load("libGeom");
  // g4root
  gSystem->Load("libG4root");

  cout << "Loading libraries ... finished" << endl;
}

void g4libs_global()
{
// Loads G4 global libraries and G4root library. 
// external packages: CLHEP, used by G4
// ---

  cout << "Loading Geant4 global libraries ..." << endl;
 
   // CLHEP
  gSystem->Load("$(CLHEP_BASE_DIR)/lib/libCLHEP");

  // Geant4
  gSystem->Load("libG4global");
  gSystem->Load("libG4graphics_reps");
  gSystem->Load("libG4intercoms");
  gSystem->Load("libG4materials");
  gSystem->Load("libG4geometry");
  // geom
  gSystem->Load("libGeom");
  // g4root
  gSystem->Load("libG4root");
  cout << "Loading libraries ... finished" << endl;
}

