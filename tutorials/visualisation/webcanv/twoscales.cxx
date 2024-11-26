/// \file
/// \ingroup tutorial_webcanv
/// \notebook -js
/// Two fully interactive scales in web canvas.
///
/// Shows two scales drawing for X or Y axis
/// Several objects can be drawn on the frame and one can select which axis is used for drawing
/// Y+ means that Y drawn on right frame side, X+ - X will be drawn on top frame side
/// Several objects can be add to the frame and associated with normal or opposite axis drawing -
/// like using "same,Y+" option
///
/// Functionality available only in web-based graphics
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \author Sergey Linev

void twoscales()
{
   auto hpe = new TH1F("hpe", "Use left and right side for Y scale drawing", 100, 0, 10);
   hpe->GetYaxis()->SetTitle("Expo");

   auto hpg = new TH1F("hpg", "Gaus distribution", 100, 0, 10);
   hpg->GetYaxis()->SetTitle("#color[2]{Gaus1} / #color[3]{Gaus2}");
   hpg->GetYaxis()->SetAxisColor(kRed);
   hpg->SetLineColor(kRed);

   auto hpg2 = new TH1F("hpg2", "Narrow gaus distribution", 100, 0, 10);
   hpg2->SetLineColor(kGreen);

   for (int i = 0; i < 25000; i++) {
      hpe->Fill(gRandom->Exp(1.));
      hpg->Fill(gRandom->Gaus(4, 1.));
      if (i % 10 == 0)
         hpg2->Fill(gRandom->Gaus(8, 0.25));
   }

   auto gr1 = new TGraph(1000);
   auto gr2 = new TGraph(10000);
   for (int i = 0; i < 10000; i++) {
      auto x = 20. + i / 100.;
      if ((i >= 2000) && (i < 3000))
         gr1->SetPoint(i - 2000, x, 1.5 + TMath::Sin(x));
      gr2->SetPoint(i, x, 3.5 + TMath::Sin(x));
   }

   gr1->SetMinimum(0);
   gr1->SetMaximum(5);
   gr1->SetTitle("Tow graphs sharing same Y scale, but using different X scales");
   gr1->GetXaxis()->SetTitle("Graph1");

   gr2->SetLineColor(kBlue);
   gr2->GetXaxis()->SetAxisColor(kBlue);
   gr2->GetXaxis()->SetTitle("Graph2");
   gr2->GetXaxis()->SetTitleColor(kBlue);

   gStyle->SetStatX(0.88);

   auto c1 = new TCanvas("c1", "Twoscales example", 1200, 800);

   if (!gROOT->IsBatch() && !c1->IsWeb())
      ::Warning("twoscales.cxx", "macro may not work without enabling web-based canvas");

   c1->Divide(1, 2);

   c1->GetPad(1)->Add(hpe);             // normal drawing
   c1->GetPad(1)->Add(hpg, "Y+");       // draw independent Y axis on right side
   c1->GetPad(1)->Add(hpg2, "same,Y+"); // use Y drawn on right side

   c1->GetPad(2)->SetTopMargin(0.2); // need more space on top
   c1->GetPad(2)->Add(gr1, "AL");    // draw as line
   c1->GetPad(2)->Add(gr2, "AL,X+"); // draw as line with independent X axis on top of the frame
}
