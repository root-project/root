//Author: Timur Pocheptsov, 25/09/2012
//This demo shows how to use transparency.
//On MacOS X you can see the transparency in a canvas,
//you can save canvas contents as pdf/png
//(and thus you'll have an image with transparency on every platform).

//Includes for ACLiC.
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TError.h"
#include "TH1F.h"

//Aux. functions.
#include "customcolors.h"

void transp()
{
   //This demo shows, how to use transparency.
   //On MacOS X you can see the transparency in a canvas,
   //you can save canvas contents as pdf/png
   //(and thus you'll have an image with transparency on every platform.

   //Unfortunately, these transparent colors can
   //not be saved with a histogram object in a file,
   //since ROOT just save color indices and our transparent
   //colors were created/added "on the fly".

   Color_t idx[2] = {};
   if (FindFreeCustomColorIndices(2, idx) != 2) {
      ::Error("transp", "failed to allocate custom colors");
      return;
   }

   TCanvas * const cnv = new TCanvas("trasnparency", "transparency demo", 600, 400);
   
   //After we created a canvas, gVirtualX in principle should be initialized
   //and we can check its type:
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Warning("transp", "this demo requires ROOT built for OS X with --enable-cocoa");
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