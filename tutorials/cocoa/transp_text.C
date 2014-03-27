//This macro is based on labels1.C by Rene Brun.
//Updated by Timur Pocheptsov to use transparent text (12/10/2012).

//Includes for ACLiC.
#include "TVirtualX.h"
#include "TPaveText.h"
#include "TCanvas.h"
#include "TRandom.h"

#include "TError.h"
#include "TColor.h"
#include "TH1F.h"

//Includes for ACLiC:
#include "customcolors.h"

void transp_text()
{
   Color_t colors[2] = {};
   if (FindFreeCustomColorIndices(2, colors) != 2) {
      ::Error("transp_text", "failed to allocate new custom colors");
      return;
   }

   TCanvas * const c1 = new TCanvas("transparent text","transparent text demo", 10, 10, 900, 500);
   //After we created a canvas, gVirtualX in principle should be initialized
   //and we can check its type:
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Warning("transp_text",
                "This macro requires ROOT built for OS X with --enable-cocoa");
   }
   
   c1->SetGrid();
   c1->SetBottomMargin(0.15);

   const Int_t nx = 20;
   const char *people[nx] = {"Jean","Pierre","Marie","Odile",
                             "Sebastien","Fons","Rene","Nicolas","Xavier","Greg",
                             "Bjarne","Anton","Otto","Eddy","Peter","Pasha",
                             "Philippe","Suzanne","Jeff","Valery"};

   TH1F * const h = new TH1F("h","test", nx, 0., nx);
   h->SetFillColor(38);
   
   for (Int_t i = 0; i < 5000; ++i)
      h->Fill(gRandom->Gaus(0.5 * nx, 0.2 * nx));

   h->SetStats(0);
   
   for (Int_t i = 1; i <= nx; ++i)
      h->GetXaxis()->SetBinLabel(i, people[i - 1]);

   h->Draw();
   
   TPaveText * const pt = new TPaveText(0.3, 0.3, 0.98, 0.98, "brNDC");
   
   //Create special transparent colors for both pavetext fill color and text color.
   new TColor(colors[0], 0.8, 0.8, 0.8, "transparent_gray", 0.85);
   pt->SetFillColor(colors[0]);
   //Add new color with index 1042.
   new TColor(colors[1], 0., 0., 0., "transparent_black", 0.5);
   pt->SetTextColor(colors[1]);
   pt->SetTextSize(0.5);
   pt->SetTextAlign(12);

   pt->AddText("Hello");
   pt->Draw();
}
