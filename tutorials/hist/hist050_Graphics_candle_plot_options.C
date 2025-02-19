/// \file
/// \ingroup tutorial_hist
/// \notebook
/// \preview Example showing how to combine the various candle plot options.
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \date December 2017
/// \author Georg Troska

void hist050_Graphics_candle_plot_options()
{
   TCanvas *c1 = new TCanvas("c1", "Candle Presets", 800, 600);
   c1->Divide(3, 2);

   TRandom *rng = new TRandom();
   TH2I *h1 = new TH2I("h1", "Sin", 18, 0, 360, 300, -1.5, 1.5);
   h1->GetXaxis()->SetTitle("Deg");
   float myRand;
   for (int i = 0; i < 360; i += 10) {
      for (int j = 0; j < 100; j++) {
         myRand = rng->Gaus(sin(i * 3.14 / 180), 0.2);
         h1->Fill(i, myRand);
      }
   }
   for (int i = 1; i < 7; i++) {
      c1->cd(i);
      TString str = TString::Format("candlex%d", i);
      TH2I *myhist = (TH2I *)h1->DrawCopy(str);
      myhist->SetTitle(str);
   }

   TCanvas *c2 = new TCanvas("c2", "Candle Individual", 800, 600);
   c2->Divide(4, 4);
   char myopt[16][8] = {"0",   "1",    "11",   "21",    "31",     "30",     "111",   "311",
                        "301", "1111", "2321", "12111", "112111", "212111", "312111"};
   for (int i = 0; i < 15; i++) {
      c2->cd(i + 1);
      TString str = TString::Format("candlex(%s)", myopt[i]);
      TH2I *myhist = (TH2I *)h1->DrawCopy(str);
      myhist->SetTitle(str);
   }
}
