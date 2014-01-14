//This macro shows how to create and use linear gradients to fill
//a histogram or a pad.
//Requires OS X and ROOT configured with --enable-cocoa.

//Includes for ACLiC (cling does not need them).
#include "TColorGradient.h"
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TError.h"
#include "TH1F.h"

//Aux. functions for tutorials/cocoa.
#include "customcolor.h"

//______________________________________________________________________
void grad()
{
   //1. Try to 'allocate' free indices for our custom colors.
   //We can use hard-coded indices like 1001, 1002, 1003, ... but
   //I prefer to find free indices in the ROOT's color table
   //to avoid possible conflicts with other tutorials.
   Int_t colorIndices[5] = {};
   if (ROOT::CocoaTutorials::FindFreeCustomColorIndices(colorIndices) != 5) {
      Error("grad", "failed to create new custom colors");
      return;
   }

   //2. Test if we have a right GUI back-end.
   TCanvas * const cnv = new TCanvas("gradient demo 1", "gradient demo 1", 100, 100, 600, 600);
   //After canvas was created, gVirtualX should be non-null.
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      Error("grad", "This macro works only on MacOS X with --enable-cocoa");
      delete cnv;
      return;
   }
   
   //3. Create custom colors.
   const Int_t frameGradient = colorIndices[2];//this gradient is a mixture of colorIndices[0] and colorIndices[1]
   //Fill color for a pad frame:
   {
      new TColor(colorIndices[0], 0.25, 0.25, 0.25, "special pad color1", 0.55);
      new TColor(colorIndices[1], 1., 1., 1., "special pad color2", 0.05);

      const Double_t locations[] = {0., 0.2, 0.8, 1.};
      const Color_t gradientIndices[4] = {colorIndices[0], colorIndices[1], colorIndices[1], colorIndices[0]};

      //Gradient for a pad's frame.
      new TColorGradient(frameGradient, TColorGradient::kGDHorizontal, 4, locations, gradientIndices);
   }

   const Int_t padGradient = colorIndices[3];
   //Fill color for a pad:
   {
      const Double_t locations[] = {0., 1.};
      const Color_t gradientIndices[4] = {38, 30};//We create a gradient from system colors.
      
      //Gradient for a pad.
      new TColorGradient(padGradient, TColorGradient::kGDVertical, 2, locations, gradientIndices);
   }
   
   const Int_t histGradient = colorIndices[4];
   //Fill color for a histogram:
   {
      const Color_t gradientIndices[3] = {kRed, kOrange, kYellow};
      const Double_t lengths[3] = {0., 0.5, 1.};
      
      //Gradient for a histogram.
      new TColorGradient(histGradient, TColorGradient::kGDVertical, 3, lengths, gradientIndices);
   }

   cnv->SetFillColor(padGradient);
   cnv->SetFrameFillColor(frameGradient);
   
   TH1F * const hist = new TH1F("a1", "b1", 20, -3., 3.);
   hist->SetFillColor(histGradient);
   hist->FillRandom("gaus", 100000);
   hist->Draw();

   cnv->Update();
}
