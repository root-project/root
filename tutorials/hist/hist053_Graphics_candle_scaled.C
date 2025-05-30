/// \file
/// \ingroup tutorial_hist
/// \notebook
/// \preview Candle Scaled, illustrates what scaling does on candle and violin charts.
/// Please try to modify the static functions SetScaledCandle and SetScaledViolin
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \date February 2023
/// \author Georg Troska

void hist053_Graphics_candle_scaled()
{
   TCanvas *c1 = new TCanvas("c1", "TCandle Scaled", 800, 600);
   c1->Divide(2, 2);
   TH2I *h1 = new TH2I("h1", "GausXY", 20, -5, 5, 100, -5, 5);
   TH2I *h3 = new TH2I("h3", "GausXY", 100, -5, 5, 20, -5, 5);

   for (int j = 0; j < 100000; j++) {
      auto myRand1 = gRandom->Gaus(0, 1);
      auto myRand2 = gRandom->Gaus(0, 1);
      h1->Fill(myRand1, myRand2);
      h3->Fill(myRand1, myRand2);
   }

   c1->cd(1);

   TCandle::SetScaledCandle(true); /* This is a global option for all existing candles, default is false */

   h1->SetTitle("CandleX scaled");
   h1->DrawCopy("candleX2");
   c1->cd(2);

   h3->SetTitle("CandleY scaled");
   h3->DrawCopy("candleY2");

   TCandle::SetScaledViolin(false); /* This is a global option for all existing violin, default is true */
   TH2I *h2 = (TH2I *)h1->Clone();
   h2->SetFillStyle(0);
   h2->SetFillColor(kGray + 2);
   h2->SetLineColor(kBlue);
   TH2I *h4 = (TH2I *)h3->Clone();
   h4->SetFillStyle(0);
   h4->SetFillColor(kGray + 2);
   h4->SetLineColor(kBlue);

   c1->cd(3);
   h2->SetTitle("ViolinX unscaled");
   h2->DrawCopy("ViolinX");
   c1->cd(4);
   h4->SetTitle("ViolinY unscaled");
   h4->DrawCopy("ViolinY");
}
