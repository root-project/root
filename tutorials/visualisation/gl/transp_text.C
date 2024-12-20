/// \file
/// \ingroup tutorial_gl
/// This macro is based on labels1.C by Rene Brun.
/// Updated by Timur Pocheptsov to use transparent text.
/// The macro requires OpenGL or Web-based canvas
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \authors Timur Pocheptsov, Sergey Linev

// Includes for ACLiC (cling does not need them).
#include "TPaveText.h"
#include "TCanvas.h"
#include "TRandom.h"
#include "TError.h"
#include "TColor.h"
#include "TStyle.h"
#include "TH1F.h"

void transp_text(bool gl = true)
{
   // 1. Create special transparent colors for both pavetext fill color and text color.
   auto grayColorIndex = TColor::GetColor((Float_t)0.8, 0.8, 0.8, 0.85);
   auto blackColorIndex = TColor::GetColor((Float_t)0., 0., 0., 0.5);

   // 2. Create a TCanvas.
   gStyle->SetCanvasPreferGL(gl);

   auto c1 = new TCanvas("transp_text", "transparent text demo", 10, 10, 900, 500);
   if (!c1->UseGL() && !c1->IsWeb())
      ::Warning("transp_text", "to use this macro you need either OpenGL or Web");

   c1->SetGrid();
   c1->SetBottomMargin(0.15);

   const Int_t nx = 20;
   const char *people[nx] = {"Jean",    "Pierre", "Marie",    "Odile",   "Sebastien", "Fons",  "Rene",
                             "Nicolas", "Xavier", "Greg",     "Bjarne",  "Anton",     "Otto",  "Eddy",
                             "Peter",   "Pasha",  "Philippe", "Suzanne", "Jeff",      "Valery"};

   auto h = new TH1F("h4", "test", nx, 0, nx);

   h->SetFillColor(38);
   for (Int_t i = 0; i < 5000; ++i)
      h->Fill(gRandom->Gaus(0.5 * nx, 0.2 * nx));

   h->SetStats(false);
   for (Int_t i = 1; i <= nx; ++i)
      h->GetXaxis()->SetBinLabel(i, people[i - 1]);

   h->Draw();

   auto pt = new TPaveText(0.3, 0.3, 0.98, 0.98, "brNDC");
   // Transparent 'rectangle' with transparent text.
   pt->SetFillColor(grayColorIndex);
   pt->SetTextColor(blackColorIndex);

   pt->SetTextSize(0.5);
   pt->SetTextAlign(12);

   pt->AddText("Hello");
   pt->Draw();
}
