/// \file
/// \ingroup tutorial_hist
/// \notebook
/// \preview Example showing how a THStack with candle plot option.
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \date May 2024
/// \authors Georg Troska, Olivier Couet

void hist051_Graphics_candle_plot_stack()
{
   gStyle->SetTimeOffset(0);
   auto rng = new TRandom();
   auto dateBegin = new TDatime(2010, 1, 1, 0, 0, 0);
   auto dateEnd = new TDatime(2011, 1, 1, 0, 0, 0);
   int bins = 1000;
   auto h1 = new TH2I("h1", "Machine A", 6, dateBegin->Convert(), dateEnd->Convert(), bins, 0, 1000);
   auto h2 = new TH2I("h2", "Machine B", 6, dateBegin->Convert(), dateEnd->Convert(), bins, 0, 1000);
   auto hsum = new TH2I("h4", "Sum", 6, dateBegin->Convert(), dateEnd->Convert(), bins, 0, 1000);

   float Rand;
   for (int i = dateBegin->Convert(); i < dateEnd->Convert(); i += 86400 * 30) {
      for (int j = 0; j < 1000; j++) {
         Rand = rng->Gaus(500 + sin(i / 10000000.) * 100, 50);
         h1->Fill(i, Rand);
         hsum->Fill(i, Rand);
         Rand = rng->Gaus(500 + sin(i / 12000000.) * 100, 50);
         h2->Fill(i, Rand);
         hsum->Fill(i, Rand);
      }
   }

   h2->SetLineColor(kRed);
   hsum->SetFillColor(kGreen);
   TCanvas *c1 = new TCanvas();

   auto hs = new THStack("hs", "Machine A+B");
   hs->Add(h1);
   hs->Add(h2, "candle2");
   hs->Add(hsum, "violin1");
   hs->Draw("candle3");
   hs->GetXaxis()->SetNdivisions(410);

   gPad->SetGrid(1, 0);

   hs->GetXaxis()->SetTimeDisplay(1);
   hs->GetXaxis()->SetTimeFormat("%d/%m/%y");
   hs->GetXaxis()->SetNdivisions(-6);
   hs->GetXaxis()->SetTitle("Date [day/month/year]");
   c1->Modified();

   gPad->BuildLegend(0.75, 0.75, 0.95, 0.95, "");
}
