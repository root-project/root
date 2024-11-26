/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This tutorial illustrates the usage of the standard scaler as preprocessing
/// method.
///
/// \macro_code
/// \macro_output
///
/// \date July 2019
/// \author Stefan Wunsch

using namespace TMVA::Experimental;

void tmva004_RStandardScaler()
{
   // Load data used to fit the parameters
   ROOT::RDataFrame df("TreeS", "http://root.cern/files/tmva_class_example.root");
   auto x = AsTensor<float>(df);

   // Create standard scaler and fit to data
   RStandardScaler<float> scaler;
   scaler.Fit(x);

   // Compute transformation
   auto y = scaler.Compute(x);

   // Plot first variable scaled and unscaled
   TH1F h1("h1", ";x_{4};N_{Events}", 20, -4, 4);
   TH1F h2("h2", ";x_{4};N_{Events}", 20, -4, 4);
   for (std::size_t i = 0; i < x.GetShape()[0]; i++) {
      h1.Fill(x(i, 3));
      h2.Fill(y(i, 3));
   }
   h1.SetLineWidth(2);
   h1.SetLineColor(kRed);
   h2.SetLineWidth(2);
   h2.SetLineColor(kBlue);

   gStyle->SetOptStat(0);
   auto c = new TCanvas("", "", 800, 800);
   h2.Draw("HIST");
   h1.Draw("HIST SAME");

   TLegend legend(0.7, 0.7, 0.89, 0.89);
   legend.SetBorderSize(0);
   legend.AddEntry("h1", "Unscaled", "l");
   legend.AddEntry("h2", "Scaled", "l");
   legend.Draw();

   c->DrawClone();
}
