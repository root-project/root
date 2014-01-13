#include "TVirtualX.h"
#include "TCanvas.h"
#include "Rtypes.h"
#include "TColor.h"
#include "TError.h"
#include "TH1F.h"

#include "freecustomcolor.h"

void transp()
{
   //This demo shows, how to use transparency.
   //On MacOS X you can see the transparency in a canvas,
   //you can save canvas contents as pdf/png
   //(and thus you'll have an image with transparency on every platform).

   //Unfortunately, these transparent colors can
   //not be saved with a histogram object in a file,
   //since ROOT just save color indices and our transparent
   //colors were created/added "on the fly".
   
   //This macro DOES NOT demonstrate a good memory management/error handling.
   
   const Int_t redIndex = ROOT::CocoaTutorials::FindFreeCustomColorIndex();
   if (redIndex == -1) {
      ::Error("transp", "no free index found for a custom red color");
      return;
   }
   
   new TColor(redIndex, 1., 0., 0., "red", 0.85);

   const Int_t greeIndex = ROOT::CocoaTutorials::FindFreeCustomColorIndex();
   if (greeIndex == -1) {
      ::Error("transp", "no free index found for a custom green color");
      return;
   }

   new TColor(greeIndex, 0., 1., 0., "green", 0.5);

   TCanvas * const cnv = new TCanvas("trasnparency", "transparency demo", 600, 400);
   
   //After we created a canvas, gVirtualX in principle should be initialized
   //and we can check its type:
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Error("transp", "You can see the transparency ONLY in a pdf or png output (\"File\"->\"Save As\" ->...)\n"
                        "To have transparency in a canvas graphics, you need MacOSX version with cocoa enabled");
   }

   TH1F * hist = new TH1F("a", "b", 10, -2., 3.);
   TH1F * hist2 = new TH1F("c", "d", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   hist->SetFillColor(redIndex);
   hist2->SetFillColor(greeIndex);
   
   cnv->cd();
   hist2->Draw();
   hist->Draw("SAME");
}
