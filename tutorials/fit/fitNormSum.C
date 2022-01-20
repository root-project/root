/// \file
/// \ingroup tutorial_fit
/// \notebook
/// Tutorial for normalized sum of two functions
/// Here: a background exponential and a crystalball function
/// Parameters can be set:
///  1.   with the TF1 object before adding the function (for 3) and 4))
///  2.  with the TF1NormSum object (first two are the coefficients, then the non constant parameters)
///  3. with the TF1 object after adding the function
///
/// Sum can be constructed by:
///  1. by a string containing the names of the functions and/or the coefficient in front
///  2. by a string containg formulas like expo, gaus...
///  3. by the list of functions and coefficients (which are 1 by default)
///  4. by a std::vector for functions and coefficients
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Lorenzo Moneta

#include <Math/MinimizerOptions.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TF1NormSum.h>
#include <TFitResult.h>
#include <TH1.h>
#include <TLatex.h>
#include <TMath.h>
#include <TStopwatch.h>
#include <TStyle.h>

void fitNormSum()
{
   const int nsig = 5.E4;
   const int nbkg = 1.e6;
   int nEvents = nsig + nbkg;
   int nBins = 1e3;

   double signal_mean = 3;
   TF1 *f_cb = new TF1("MyCrystalBall", "crystalball", -5., 5.);
   TF1 *f_exp = new TF1("MyExponential", "expo", -5., 5.);

   // I.:
   f_exp->SetParameters(1., -0.3);
   f_cb->SetParameters(1, signal_mean, 0.3, 2, 1.5);

   // CONSTRUCTION OF THE TF1NORMSUM OBJECT ........................................
   // 1) :
   TF1NormSum *fnorm_exp_cb = new TF1NormSum(f_cb, f_exp, nsig, nbkg);
   // 4) :

   TF1 *f_sum = new TF1("fsum", *fnorm_exp_cb, -5., 5., fnorm_exp_cb->GetNpar());

   // III.:
   f_sum->SetParameters(fnorm_exp_cb->GetParameters().data());
   f_sum->SetParName(1, "NBackground");
   f_sum->SetParName(0, "NSignal");
   for (int i = 2; i < f_sum->GetNpar(); ++i)
      f_sum->SetParName(i, fnorm_exp_cb->GetParName(i));

   // GENERATE HISTOGRAM TO FIT ..............................................................
   TStopwatch w;
   w.Start();
   TH1D *h_sum = new TH1D("h_ExpCB", "Exponential Bkg + CrystalBall function", nBins, -5., 5.);
   h_sum->FillRandom("fsum", nEvents);
   printf("Time to generate %d events:  ", nEvents);
   w.Print();

   // need to scale histogram with width since we are fitting a density
   h_sum->Sumw2();
   h_sum->Scale(1., "width");

   // fit - use Minuit2 if available
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
   new TCanvas("Fit", "Fit", 800, 1000);
   // do a least-square fit of the spectrum
   auto result = h_sum->Fit("fsum", "SQ");
   result->Print();
   h_sum->Draw();
   printf("Time to fit using ROOT TF1Normsum: ");
   w.Print();

   // test if parameters are fine
   std::vector<double> pref = {nsig, nbkg, signal_mean};
   for (unsigned int i = 0; i < pref.size(); ++i) {
      if (!TMath::AreEqualAbs(pref[i], f_sum->GetParameter(i), f_sum->GetParError(i) * 10.))
         Error("testFitNormSum", "Difference found in fitted %s - difference is %g sigma", f_sum->GetParName(i),
               (f_sum->GetParameter(i) - pref[i]) / f_sum->GetParError(i));
   }

   gStyle->SetOptStat(0);
   // add parameters
   auto t1 = new TLatex(
      -2.5, 300000, TString::Format("%s = %8.0f #pm %4.0f", "NSignal", f_sum->GetParameter(0), f_sum->GetParError(0)));
   auto t2 = new TLatex(
      -2.5, 270000, TString::Format("%s = %8.0f #pm %4.0f", "Nbackgr", f_sum->GetParameter(1), f_sum->GetParError(1)));
   t1->Draw();
   t2->Draw();
}
