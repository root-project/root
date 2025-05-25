/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// \preview Histograms filled and drawn in a loop.
/// Simple example illustrating how to use the C++ interpreter
/// to fill histograms in a loop and show the graphics results
///
/// \macro_image
/// \macro_code
///
/// \date November 2024
/// \author Rene Brun

void hist007_TH1_liveupdate()
{
   TCanvas *c1 = new TCanvas("c1", "Live update of histograms", 200, 10, 600, 400);
   c1->SetGrid();

   // Create some histograms.
   auto *total = new TH1D("total", "This is the total distribution", 100, -4, 4);
   auto *main = new TH1D("main", "Main contributor", 100, -4, 4);
   auto *s1 = new TH1D("s1", "This is the first signal", 100, -4, 4);
   auto *s2 = new TH1D("s2", "This is the second signal", 100, -4, 4);
   total->Sumw2(); // store the sum of squares of weights
   // set some style properties
   total->SetMarkerStyle(21); // shape of the markers (\see EMarkerStyle)
   total->SetMarkerSize(0.7);
   main->SetFillColor(16);
   s1->SetFillColor(42);
   s2->SetFillColor(46);
   TSlider *slider = nullptr;

   // Fill histograms randomly
   TRandom3 rng;
   const int kUPDATE = 500;
   for (int i = 0; i < 10000; i++) {
      float xmain = rng.Gaus(-1, 1.5);
      float xs1 = rng.Gaus(-0.5, 0.5);
      float xs2 = rng.Landau(1, 0.15);
      main->Fill(xmain);
      s1->Fill(xs1, 0.3);
      s2->Fill(xs2, 0.2);
      total->Fill(xmain);
      total->Fill(xs1, 0.3);
      total->Fill(xs2, 0.2);
      if (i && (i % kUPDATE) == 0) {
         if (i == kUPDATE) {
            total->Draw("e1p");
            main->Draw("same");
            s1->Draw("same");
            s2->Draw("same");
            c1->Update();
            slider = new TSlider("slider", "test", 4.2, 0, 4.6, total->GetMaximum(), 38);
            slider->SetFillColor(46);
         }
         if (slider)
            slider->SetRange(0., 1. * i / 10000.);
         c1->Modified();
         c1->Update();
      }
   }
   slider->SetRange(0., 1.);
   c1->Modified();
}
