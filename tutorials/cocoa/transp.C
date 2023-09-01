/// \file
/// \ingroup tutorial_cocoa
/// This demo shows how to use transparency.
/// On MacOS X you can see the transparency in a canvas,
/// you can save canvas contents as pdf/png
/// (and thus you'll have an image with transparency on every platform).
///
/// \macro_code
///
/// \author Timur Pocheptsov

//Includes for ACLiC (cling does not need them).
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TError.h"
#include "TH1F.h"

//Aux. functions for tutorials/cocoa.
#include "customcolor.h"

void transp()
{
   //1. Try to find free indices for our custom colors.
   //We can use hard-coded indices like 1001, 1002, 1003, ... but
   //I prefer to find free indices in a ROOT's color table
   //to avoid possible conflicts with other tutorials.
   Color_t indices[2] = {};
   if (ROOT::CocoaTutorials::FindFreeCustomColorIndices(indices) != 2) {
      ::Error("transp", "failed to create new custom colors");
      return;
   }

   //2. Now that we have indices, create our custom colors.
   const Color_t redIndex = indices[0], greeIndex = indices[1];

   new TColor(redIndex, 1., 0., 0., "red", 0.85);
   new TColor(greeIndex, 0., 1., 0., "green", 0.5);

   //3. Test that back-end is TGCocoa.
   TCanvas * const cnv = new TCanvas("transparency", "transparency demo", 600, 400);
   //After we created a canvas, gVirtualX in principle should be initialized
   //and we can check its type:
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      Warning("transp", "You can see the transparency ONLY in a pdf or png output (\"File\"->\"Save As\" ->...)\n"
                        "To have transparency in a canvas graphics, you need MacOSX version with cocoa enabled");
   }

   TH1F * const hist = new TH1F("a5", "b5", 10, -2., 3.);
   TH1F * const hist2 = new TH1F("c6", "d6", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   hist->SetFillColor(redIndex);
   hist2->SetFillColor(greeIndex);

   cnv->cd();
   hist2->Draw();
   hist->Draw("SAME");
}
