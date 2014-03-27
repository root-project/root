//Author: Timur Pocheptsov, 19/03/2014
//Requires OpenGL: either set OpenGL.CanvasPreferGL to 1
//in the $ROOTSYS/etc/system.rootrc,
//or use gStyle->SetCanvasPreferGL(kTRUE).

//Features:
//1. Linear gradients
//2. Semitransparent colors.

//Includes for ACLiC:
#include "TColorGradient.h"
#include "TCanvas.h"
#include "TError.h"
#include "TStyle.h"
#include "TH1F.h"

#include "customcolors.h"

typedef TColorGradient::Point point_type;

void grad2()
{
   Color_t customIdx[4] = {};
   if (FindFreeCustomColorIndices(4, customIdx) != 4) {
      ::Error("grad2", "failed to allocate custom colors");
      return;
   }

   gStyle->SetCanvasPreferGL(kTRUE);
   
   TCanvas * const cnv = new TCanvas("gradient test 2", "gradient_test2", 100, 100, 800, 600);
   if (!cnv->UseGL()) {
      ::Error("grad2", "This macro requires OpenGL");
      delete cnv;
      return;
   }

   TH1F * const hist = new TH1F("hg2a", "hg2a", 10, -2., 3.);
   TH1F * const hist2 = new TH1F("hg2b", "hg2b", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   new TColor(customIdx[0], 1., 0., 0., "red", 0.5);

   const Double_t locations[] = {0., 1.};
   const Color_t idx1[] = {customIdx[0], kOrange};
   //Gradient from ROOT's kOrange and my own semi-transparent red color.
   TLinearGradient * const grad1 = new TLinearGradient(customIdx[1], 2, locations, idx1);
   const point_type start(0., 0.);
   const point_type end(0., 1.);
   grad1->SetStartEnd(start, end);
   
   hist->SetFillColor(customIdx[1]);
   
   new TColor(customIdx[2], 0., 1., 0., "green", 0.5);
   const Color_t idx2[] = {customIdx[2], kBlue};
   //Gradient from ROOT's kBlue and my own semi-transparent green color.
   TLinearGradient * const grad2 = new TLinearGradient(customIdx[3], 2, locations, idx2);
   grad2->SetStartEnd(start, end);
   hist2->SetFillColor(customIdx[3]);
   
   hist2->Draw();
   hist->Draw("SAME");
}
