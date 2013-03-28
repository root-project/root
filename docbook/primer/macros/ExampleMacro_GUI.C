/*
 This piece of code demonstrates how a root macro is used as a standalone
    application with full acces the grapical user interface (GUI) of ROOT   */

// include ALL header files needed
#ifndef __CINT__
#include "TROOT.h"
#include "TApplication.h"
#include "TBrowser.h"
#include "TFile.h"    
#include "TH1F.h"     
#include "TCanvas.h"  
#include "TMath.h"
#endif
// eventually, include some additoinal C or C++ libraries
#include <math.h>
 
//     ==>>  put the code of your macro here 
void ExampleMacro_GUI() { 
  // Create a histogram, fill it with random gaussian numbers
  TH1F *h = new TH1F ("h", "example histogram", 100, -5.,5.);
  h->FillRandom("gaus",1000);

  // draw the histogram 
  h->DrawClone();

/* - Create a new ROOT file for output
   - Note that this file may contain any kind of ROOT objects, histograms,
     pictures, graphics objects etc. 
   - the new file is now becoming the current directory */
  TFile *f1 = new TFile("ExampleMacro.root","RECREATE","ExampleMacro");

  // write Histogram to current directory (i.e. the file just opened)
  h->Write();

  // Close the file. 
  //   (You may inspect your histogram in the file using the TBrowser class)
  f1->Close();
}

// the "dressing" code for a stand-alone ROOT application starts here
#ifndef __CINT__
void StandaloneApplication(int argc, char** argv) {
  // ==>> here the ROOT macro is called
  ExampleMacro_GUI();
}

// This is the standard main of C++ starting a ROOT application
int main(int argc, char** argv) {
   TApplication app("Root Application", &argc, argv);
   StandaloneApplication(app.Argc(), app.Argv());
   app.Run();
   return 0;
}
#endif
