//This demo shows how to use transparency.

//Includes for ACLiC (cling does not need them).
#include "TCanvas.h"
#include "TColor.h"
#include "TError.h"
#include "TStyle.h"
#include "TH1F.h"

//Aux. functions for tutorials/gl.
#include "customcolorgl.h"

void transp()
{
   //1. Try to find free indices for our custom colors.
   //We can use hard-coded indices like 1001, 1002, 1003, ... but
   //I prefer to find free indices in a ROOT's color table
   //to avoid possible conflicts with other tutorials.
   Int_t indices[2] = {};
   if (ROOT::GLTutorials::FindFreeCustomColorIndices(indices) != 2) {
      ::Error("transp", "failed to create new custom colors");
      return;
   }

   //2. Now that we have indices, create our custom colors.
   const Int_t redIndex = indices[0], greeIndex = indices[1];

   new TColor(redIndex, 1., 0., 0., "red", 0.85);
   new TColor(greeIndex, 0., 1., 0., "green", 0.5);

   gStyle->SetCanvasPreferGL(kTRUE);
   TCanvas * const cnv = new TCanvas("trasnparency", "transparency demo", 600, 400);

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
