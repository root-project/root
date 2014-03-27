//Author: Timur Pocheptsov, 19/03/2014

//Includes for ACLiC.
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TError.h"
#include "TH1F.h"

//Aux. functions.
#include "customcolors.h"

void transp()
{
   //This demo shows, how to use transparency with gl-pad.
   //To enable OpenGL you can either call gStyle->SetCanvasPreferGL(kTRUE)
   //or set OpenGL.CanvasPreferGL to 1 in a $ROOTSYS/etc/system.rootrc.

   Color_t idx[2] = {};
   if (FindFreeCustomColorIndices(2, idx) != 2) {
      ::Error("transp", "failed to allocate custom colors");
      return;
   }

   gStyle->SetCanvasPreferGL(kTRUE);

   TCanvas * const cnv = new TCanvas("trasnparency", "transparency demo", 600, 400);
   if (!cnv->UseGL()) {
      ::Error("transp", "this macro requires OpenGL");
      delete cnv;
      return;
   }
   
   TH1F * const hist = new TH1F("a", "b", 10, -2., 3.);
   TH1F * const hist2 = new TH1F("c", "d", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   //Add new color with index 1001.
   new TColor(idx[0], 1., 0., 0., "red", 0.85);
   hist->SetFillColor(idx[0]);
   
   //Add new color with index 1002.
   new TColor(idx[1], 0., 1., 0., "green", 0.5);
   hist2->SetFillColor(idx[1]);
   
   cnv->cd();
   hist2->Draw();
   hist->Draw("SAME");
}
