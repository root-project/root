//Gradient fill with transparency and "SAME" option.
//Requires OS X and ROOT configured with --enable-cocoa.

//Includes for ACLiC (cling does not need them).
#include "TColorGradient.h"
#include "TCanvas.h"
#include "TError.h"
#include "TColor.h"
#include "TH1F.h"

//Aux. functions for tutorials/cocoa.
#include "customcolor.h"

void grad2()
{
   //1. 'Allocate' free indices for our custom colors.
   //We can use hard-coded indices like 1001, 1002, 1003 ... but
   //I prefer to find free indices in the ROOT's color table
   //to avoid possible conflicts with other tutorials.
   Int_t freeIndices[4] = {};
   if (ROOT::CocoaTutorials::FindFreeCustomColorIndices(freeIndices) != 4) {
      Error("grad2", "can not allocate new custom colors");
      return;
   }

   //'Aliases' (instead of freeIndices[someIndex])
   const Int_t customRed = freeIndices[0], grad1 = freeIndices[1];
   const Int_t customGreen = freeIndices[2], grad2 = freeIndices[3];

   //2. Check that we are ROOT with Cocoa back-end enabled.
   TCanvas * const cnv = new TCanvas("gradiend demo 2", "gradient demo 2", 100, 100, 800, 600);
   //After canvas was created, gVirtualX should be non-null.
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      Error("grad2", "This macro works only on OS X with --enable-cocoa");
      //Unfortunately, we can not remove the colors we added :(
      delete cnv;
      return;
   }

   //3. Now that we 'allocated' all indices and checked back-end's type, let's create our custom colors colors:
   //   a) Custom semi-transparent red.
   new TColor(customRed, 1., 0., 0., "red", 0.5);

   //   b) Gradient (from our semi-transparent red to ROOT's kOrange).
   const Double_t locations[] = {0., 1.};
   const Color_t idx1[] = {kOrange, customRed};
   new TColorGradient(grad1, TColorGradient::kGDVertical, 2, locations, idx1);

   //   c) Custom semi-transparent green.
   new TColor(customGreen, 0., 1., 0., "green", 0.5);

   //   d) Gradient from ROOT's kBlue to our custom green.
   const Color_t idx2[] = {kBlue, customGreen};
   new TColorGradient(grad2, TColorGradient::kGDVertical, 2, locations, idx2);

   TH1F * hist = new TH1F("a2", "b2", 10, -2., 3.);
   TH1F * hist2 = new TH1F("c3", "d3", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   hist->SetFillColor(grad1);
   hist2->SetFillColor(grad2);

   hist2->Draw();
   hist->Draw("SAME");
}
