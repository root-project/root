//This macro is based on labels1.C by Rene Brun.
//Updated by Timur Pocheptsov to use transparent text.

//All these includes (except freecustomcolor.h) are to make
//this macro 'ACLiCable'.

#include "TVirtualX.h"
#include "TPaveText.h"
#include "TCanvas.h"
#include "TRandom.h"
#include "Rtypes.h"
#include "TError.h"
#include "TColor.h"
#include "TH1F.h"

#include "freecustomcolor.h"

void transp_text()
{
   //This macro only shows how to use transparency with a TPaveText.
   //In no way it's a good example of a right memory management
   //or an error handling.
   //Requires OS X and ROOT configured with --enable-cocoa.

   if (!gRandom) {
      ::Error("transp_text", "gRandom is null");
      return;
   }

   //1. Let's check, if we can create custom colors first:
   const Int_t grayColorIndex = ROOT::CocoaTutorials::FindFreeCustomColorIndex();
   if (grayColorIndex == -1) {
      ::Error("transp_text", "failed to create a custom color - no index found for transparent gray");
      return;
   }

   //Create special transparent colors for both pavetext fill color and text color.
   new TColor(grayColorIndex, 0.8, 0.8, 0.8, "transparent_gray", 0.85);
   
   const Int_t blackColorIndex = ROOT::CocoaTutorials::FindFreeCustomColorIndex();
   if (blackColorIndex == -1) {
      ::Error("transp_text", "failed to create a custom color - no index found for transparent black");
      return;
   }

   new TColor(blackColorIndex, 0., 0., 0., "transparent_black", 0.5);

   //Create a TCanvas first to force gVirtualX initialization.
   TCanvas * const c1 = new TCanvas("c1","transparent text demo", 10, 10, 900, 500);
   //We can check gVirtualX (its type):
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Warning("transt_text", "You can see the transparency ONLY in a pdf or png output (\"File\"->\"Save As\" ->...)\n"
                               "To have transparency in a canvas graphics, you need OS X version with cocoa enabled");
   }

   const Int_t nx = 20;
   const char *people[nx] = {"Jean","Pierre","Marie","Odile",
      "Sebastien","Fons","Rene","Nicolas","Xavier","Greg",
      "Bjarne","Anton","Otto","Eddy","Peter","Pasha",
      "Philippe","Suzanne","Jeff","Valery"};

   c1->SetGrid();
   c1->SetBottomMargin(0.15);

   TH1F * const h = new TH1F("h", "test", nx, 0, nx);

   h->SetFillColor(38);
   for (Int_t i = 0; i < 5000; ++i)
      h->Fill(gRandom->Gaus(0.5 * nx, 0.2 * nx));

   h->SetStats(0);
   for (Int_t i = 1; i <= nx; ++i)
      h->GetXaxis()->SetBinLabel(i, people[i - 1]);

   h->Draw();
   
   TPaveText * const pt = new TPaveText(0.3, 0.3, 0.98, 0.98, "brNDC");
   //Transparent 'rectangle' with transparent text.
   pt->SetFillColor(grayColorIndex);
   pt->SetTextColor(blackColorIndex);

   pt->SetTextSize(0.5);
   pt->SetTextAlign(12);
   
   pt->AddText("Hello");
   pt->Draw();
}
