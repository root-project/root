//Author: Timur Pocheptsov, 25/09/2012 (?)
//This macro requires OS X and ROOT
//compiled with --enable-cocoa to run.

//Features:
//1. Linear gradients
//2. Semitransparent colors.

//Includes for ACLiC:
#include "TColorGradient.h"
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TError.h"
#include "TH1F.h"

//Aux. functions.
#include "customcolors.h"

typedef TColorGradient::Point point_type;

void grad2()
{
   //Gradient fill with transparency and "SAME" option.
   //This macro works ONLY on MacOS X with --enable-cocoa.
   
   Color_t idx[4] = {};
   if (FindFreeCustomColorIndices(4, idx) != 4) {
      ::Error("grad2", "failed to allocate custom colors");
      return;
   }
   
   TCanvas * const cnv = new TCanvas("gradient test 2", "gradient_test2", 100, 100, 800, 600);
   //After canvas was created, gVirtualX should be non-null.
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Error("grad2", "This macro works only on MacOS X with --enable-cocoa");
      delete cnv;
      return;
   }

   TH1F * const hist = new TH1F("hg2a", "hg2a", 10, -2., 3.);
   TH1F * const hist2 = new TH1F("hg2b", "hg2b", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   const Double_t locations[] = {0., 1.};
   new TColor(idx[0], 1., 0., 0., "red", 0.5);
   const Color_t idx1[] = {idx[0], kOrange};
   //Gradient from ROOT's kOrange and my own semi-transparent red color.
   TLinearGradient * const grad1 = new TLinearGradient(idx[1], 2, locations, idx1);
   const point_type start(0., 0.);
   const point_type end(0., 1.);
   
   grad1->SetStartEnd(start, end);
   
   hist->SetFillColor(idx[1]);
   
   new TColor(idx[2], 0., 1., 0., "green", 0.5);
   const Color_t idx2[] = {idx[2], kBlue};
   //Gradient from ROOT's kBlue and my own semi-transparent green color.
   TLinearGradient * const grad2 = new TLinearGradient(idx[3], 2, locations, idx2);
   grad2->SetStartEnd(start, end);
   hist2->SetFillColor(idx[3]);
   
   hist2->Draw();
   hist->Draw("SAME");
}
