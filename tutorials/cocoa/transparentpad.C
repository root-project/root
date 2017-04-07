/// \file
/// \ingroup tutorial_cocoa
/// This macro demonstrates semi-transparent pads.
/// Requires OS X and ROOT configured with --enable-cocoa.
///
/// \macro_code
///
/// \author Timur Pocheptsov

//Includes for ACLiC (cling does not need them).
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TError.h"
#include "TColor.h"
#include "TH1F.h"

//Aux. functions for tutorials/cocoa.
#include "customcolor.h"

void transparentpad()
{
   //1. Try to 'allocate' free indices for our custom colors -
   //we can use hard-coded indices like 1001, 1002, 1003 ... but
   //I prefer to find free indices in a ROOT's color table
   //to avoid possible conflicts with other tutorials.
   Color_t indices[3] = {};
   if (ROOT::CocoaTutorials::FindFreeCustomColorIndices(indices) != 3) {
      ::Error("transparentpad", "failed to create new custom colors");
      return;
   }

   //2. Create a TCanvas to force a gVirtualX initialization.
   TCanvas * const c1 = new TCanvas("transparent pad","transparent pad demo", 10, 10, 900, 500);
   //We can check gVirtualX (its type):
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Warning("transparentpad", "You can see the transparency ONLY in a pdf or png output (\"File\"->\"Save As\" ->...)\n"
                                  "To have transparency in a canvas graphics, you need OS X version with cocoa enabled");
   }

   //2. Create special transparent colors.
   new TColor(indices[0], 1., 0.2, 0.2, "transparent_pink", 0.25);
   new TColor(indices[1], 0.2, 1., 0.2, "transparent_green", 0.25);
   new TColor(indices[2], 0.2, 2., 1., "transparent_blue", 0.15);

   //3. Some arbitrary histograms.
   TH1F * const h1 = new TH1F("TH1F 1", "TH1F 1", 100, -1.5, 1.5);
   h1->FillRandom("gaus");

   TH1F * const h2 = new TH1F("TH1F 2", "TH1F 2", 100, -1.5, 0.);
   h2->FillRandom("gaus");

   TH1F * const h3 = new TH1F("TH1F 3", "TH1F 3", 100, 0.5, 2.);
   h3->FillRandom("landau");

   //4. Now overlapping transparent pads.
   TPad * const pad1 = new TPad("transparent pad 1", "transparent pad 1", 0.1, 0.1, 0.7, 0.7);
   pad1->SetFillColor(indices[0]);//here's the magic!
   pad1->cd();
   h1->Draw("lego2");
   c1->cd();
   pad1->Draw();

   TPad * const pad2 = new TPad("transparent pad 2", "transparent pad 2", 0.2, 0.2, 0.8, 0.8);
   pad2->SetFillColor(indices[1]);//here's the magic!
   pad2->cd();
   h2->Draw();
   c1->cd();
   pad2->Draw();

   TPad * const pad3 = new TPad("transparent pad 3", "transparent pad 3", 0.3, 0.3, 0.9, 0.9);
   pad3->SetFillColor(indices[2]);//here's the magic!
   pad3->cd();
   h3->Draw();
   c1->cd();
   pad3->Draw();
}

