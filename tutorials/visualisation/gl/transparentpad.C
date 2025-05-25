/// \file
/// \ingroup tutorial_gl
/// This macro demonstrates semi-transparent pads.
/// Requires OpenGL or Web-based canvas.
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \authors Timur Pocheptsov, Sergey Linev

// Includes for ACLiC (cling does not need them).
#include "TCanvas.h"
#include "TStyle.h"
#include "TError.h"
#include "TColor.h"
#include "TH1F.h"

void transparentpad(bool gl = true)
{
   gStyle->SetCanvasPreferGL(gl);

   // 1. Create canvas and check if it support transparent colors
   auto c1 = new TCanvas("transparentpad", "transparent pad demo", 10, 10, 900, 500);
   if (!c1->UseGL() && !c1->IsWeb())
      ::Warning("transparentpad",
                "You can see the transparency ONLY in a pdf or png output (\"File\"->\"Save As\" ->...)\n"
                "To have transparency in a canvas graphics, you need either OpenGL or Web rendering enabled");

   // 2. Some arbitrary histograms.
   auto h1 = new TH1F("TH1F 1", "TH1F 1", 100, -1.5, 1.5);
   h1->FillRandom("gaus");

   auto h2 = new TH1F("TH1F 2", "TH1F 2", 100, -1.5, 0.);
   h2->FillRandom("gaus");

   auto h3 = new TH1F("TH1F 3", "TH1F 3", 100, 0.5, 2.);
   h3->FillRandom("landau");

   // 3. Now overlapping transparent pads.
   auto pad1 = new TPad("transparent pad 1", "transparent pad 1", 0.1, 0.1, 0.7, 0.7);
   pad1->SetFillColor(TColor::GetColor((Float_t)1., 0.2, 0.2, 0.25)); // transparent pink, here's the magic!
   c1->cd();
   pad1->Draw();
   pad1->cd();
   h1->Draw("lego2");

   auto pad2 = new TPad("transparent pad 2", "transparent pad 2", 0.2, 0.2, 0.8, 0.8);
   pad2->SetFillColor(TColor::GetColor((Float_t)0.2, 1., 0.2, 0.25)); // transparent green, here's the magic!
   c1->cd();
   pad2->Draw();
   pad2->cd();
   h2->Draw();

   auto pad3 = new TPad("transparent pad 3", "transparent pad 3", 0.3, 0.3, 0.9, 0.9);
   pad3->SetFillColor(TColor::GetColor((Float_t)0.2, 1., 1., 0.15)); // transparent blue, here's the magic!
   c1->cd();
   pad3->Draw();
   pad3->cd();
   h3->Draw();
}
