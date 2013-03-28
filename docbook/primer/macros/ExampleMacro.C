/***************************************************************************
 * --------------------------------------------------------------------------
 * Template to exercise ROOT code
 * -> Plot a normalized Gaussian with mean = 5.0 and sigma = 1.5
 *    and its derivative and integral in the range of 0 to 10
 * --------------------------------------------------------------------------
 * initial version:  21-Aug-2008  G. Quast
 *
 * modification log:
 ***************************************************************************/

/* 
 * Note that this file can be either used as a compiled program
    or as a ROOT macro.
 * If it is used as a compiled program, additional include statements
   and the definition of the main program have to be made. This is 
   not needed if the code is executed at the ROOT prompt.
                                                                            */

//#ifndef __CINT__      // These include-statements are needed if the program is
#include "TFile.h"    // run as a "stand-alone application", i.e. if it is not 
#include "TH1F.h"     // called from an interactive ROOT session.
#include "TCanvas.h"  
#include "TMath.h"
// eventually, load some C libraries
#include <math.h>

void ExampleMacro();

//______________________________________________________________________________
int main()
{
  ExampleMacro();
  return 0;
}
//#endif                 

//______________________________________________________________________________

/* 
   * From here on, the code can also be used as a macro
   * Note though, that CINT may report errors where there are none
     in C++. E.g. this happens here where CINT says that f1 is
     out of scope ...

     ==>>  put your code here 
      (remember to update the name of you Macro in the 
       lines above if you intend to comile the code)
                                                                     */
void ExampleMacro() { 
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
