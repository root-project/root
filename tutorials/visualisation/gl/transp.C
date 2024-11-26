/// \file
/// \ingroup tutorial_gl
/// This demo shows how to use transparency.
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \authors Timur Pocheptsov, Sergey Linev

// Includes for ACLiC (cling does not need them).
#include "TCanvas.h"
#include "TColor.h"
#include "TError.h"
#include "TStyle.h"
#include "TH1F.h"

void transp(bool gl = true)
{
   auto redIndex = TColor::GetColor((Float_t)1., 0., 0., 0.85);
   auto greeIndex = TColor::GetColor((Float_t)0., 1., 0., 0.5);

   gStyle->SetCanvasPreferGL(kTRUE);
   auto cnv = new TCanvas("trasnparency", "transparency demo", 600, 400);

   auto hist = new TH1F("a5", "b5", 10, -2., 3.);
   auto hist2 = new TH1F("c6", "d6", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   hist->SetFillColor(redIndex);
   hist2->SetFillColor(greeIndex);

   cnv->cd();
   hist2->Draw();
   hist->Draw("SAME");
}
