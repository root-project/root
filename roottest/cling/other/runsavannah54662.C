#include "TROOT.h"

Bool_t figsSetup=kFALSE; 
 
void SetupFigs(){

  gROOT->LoadMacro("testCompile.C+");

  figsSetup=kTRUE;
}


void runsavannah54662() {

   SetupFigs();
}
