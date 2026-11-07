/// \file
/// \ingroup tutorial_gl
/// This macro demonstrates semi-transparent pads.
///
/// One uses fill styles between 4000 and 4100 to configure objects transparency
/// On OpenGL or Mac/Cocoa or Web-based canvas pads will be drawn with transparent colors.
/// On X11 pixmap transformation performed to partially emulate pads transparency
/// Also demonstrated usage of transparent fill styles for stats box
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \authors Timur Pocheptsov, Sergey Linev

#include "TCanvas.h"
#include "TStyle.h"
#include "TError.h"
#include "TColor.h"
#include "TH1F.h"

void transparentpad(bool gl = true)
{
   gStyle->SetCanvasPreferGL(gl);

   // prevent filling of TFrame with solid color
   gStyle->SetFrameFillStyle(0);

   // 1. Create canvas and check if it support transparent colors
   auto c1 = new TCanvas("c1", "transparent pad demo", 10, 10, 900, 500);
   if (!c1->UseGL() && !c1->IsWeb() && !gVirtualX->InheritsFrom("TGCocoa"))
      ::Warning("transparentpad",
                "To have real transparency in a canvas graphics, you need either OpenGL or Mac/Cocoa or Web rendering enabled");

   // 2. Some arbitrary histograms.
   auto h1 = new TH1F("TH1F 1", "TH1F 1", 100, -1.5, 1.5);
   h1->FillRandom("gaus");

   auto h2 = new TH1F("TH1F 2", "TH1F 2", 100, -1.5, 0.);
   h2->FillRandom("gaus");

   auto h3 = new TH1F("TH1F 3", "TH1F 3", 100, 0.5, 2.);
   h3->FillRandom("landau");

   // 3. Now overlapping transparent pads.
   auto pad1 = new TPad("transparent pad 1", "transparent pad 1", 0.1, 0.1, 0.7, 0.7);
   pad1->SetFillColor(kPink);
   pad1->SetFillStyle(4040); // transparent pink, here's the magic!
   c1->Add(pad1);
   pad1->Add(h1, "lego2");

   auto pad2 = new TPad("transparent pad 2", "transparent pad 2", 0.2, 0.2, 0.8, 0.8);
   pad2->SetFillColor(kGreen);
   pad2->SetFillStyle(4035); // transparent green, here's the magic!
   c1->Add(pad2);
   pad2->Add(h2);

   auto pad3 = new TPad("transparent pad 3", "transparent pad 3", 0.3, 0.3, 0.9, 0.9);
   pad3->SetFillColor(kBlue);
   pad3->SetFillStyle(4030);   // transparent blue, here's the magic!
   c1->Add(pad3);
   pad3->Add(h3);

   c1->Update();

   auto stats2 = dynamic_cast<TPaveStats *>(h2->FindObject("stats"));
   if (stats2) {
      stats2->SetFillColor(kYellow);
      stats2->SetFillStyle(4050); // semi-transparent stats box, only with GL
   }

   auto stats3 = dynamic_cast<TPaveStats *>(h3->FindObject("stats"));
   if (stats3) {
      stats3->SetFillColor(kYellow);
      stats3->SetFillStyle(4050); // semi-transparent stats box, only with GL
   }

   // SVG or PDF or PS image always support transparent colors
   // c1->SaveAs("transparentpad.svg");

   c1->Modified();
}
