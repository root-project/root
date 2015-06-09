
#include <TMath.h>
#include <TCanvas.h>
#include <TF1NormSum.h>
#include <TF1.h>
#include <TH1.h>


using namespace std;


void fitNormSum()
{
   //***************************************************************************************************

   // Tutorial for normalized sum of two functions
   // Here: a background exponential and a crystalball function
   
   
   // Parameters can be set:
   // I.   with the TF1 object before adding the function (for 3) and 4))
   // II.  with the TF1NormSum object (first two are the coefficients, then the non constant parameters)
   // III. with the TF1 object after adding the function
   
   // Sum can be constructed by:
   // 1) by a string containing the names of the functions and/or the coefficient in front
   // 2) by a string containg formulas like expo, gaus...
   // 3) by the list of functions and coefficients (which are 1 by default)
   // 4) by a std::vector for functions and coefficients
   
   //***************************************************************************************************


   const int nsig = 5.E4;
   const int nbkg = 1.e6;
   Int_t NEvents = nsig+nbkg;
   Int_t NBins   = 1e3;

   double signal_mean = 3;
   TF1 *f_cb    = new TF1("MyCrystalBall","crystalball",-5.,5.);
   TF1 *f_exp   = new TF1("MyExponential","expo",-5.,5.);
   
   // I.:
   f_exp-> SetParameters(1.,-0.3);
   f_cb -> SetParameters(1,signal_mean,0.3,2,1.5);
   
   // CONSTRUCTION OF THE TF1NORMSUM OBJECT ........................................
   // 1) :
   TF1NormSum *fnorm_exp_cb = new TF1NormSum(f_cb,f_exp,nsig,nbkg);
   // 4) :
   
   TF1   * f_sum = new TF1("fsum", *fnorm_exp_cb, -5., 5., fnorm_exp_cb->GetNpar());
   f_sum->Draw();
   
   // III.:
   f_sum->SetParameters( fnorm_exp_cb->GetParameters().data() );
   f_sum->SetParName(1,"NBackground");
   f_sum->SetParName(0,"NSignal");
   for (int i = 2; i < f_sum->GetNpar(); ++i) 
      f_sum->SetParName(i,fnorm_exp_cb->GetParName(i) );
  
   //GENERATE HISTOGRAM TO FIT ..............................................................
   TStopwatch w;
   w.Start(); 
   TH1D *h_sum = new TH1D("h_ExpCB", "Exponential Bkg + CrystalBall function", NBins, -5., 5.);
   for (int i=0; i<NEvents; i++)
   {
      h_sum -> Fill(f_sum -> GetRandom());
   }
   printf("Time to generate %d events:  ",NEvents);
   w.Print();
   //TH1F *h_orig = new TH1F(*h_sum);
   
   // need to scale histogram with width since we are fitting a density
   h_sum -> Sumw2();
   h_sum -> Scale(1., "width");   

   //fit - use Minuit2 if available
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
   new TCanvas("Fit","Fit",800,1000);
   // do a least-square fit of the spectrum
   auto result = h_sum -> Fit("fsum","SQ");
   result->Print();
   h_sum -> Draw();
   printf("Time to fit using ROOT TF1Normsum: ");
   w.Print();

   // test if parameters are fine
   std::vector<double>  pref = {nsig, nbkg, signal_mean};
   for (unsigned int i = 0; i< pref.size(); ++i)  {
      if (!TMath::AreEqualAbs(pref[i], f_sum->GetParameter(i), f_sum->GetParError(i)*10.) )
         Error("testFitNormSum","Difference found in fitted %s - difference is %g sigma",f_sum->GetParName(i), (f_sum->GetParameter(i)-pref[i])/f_sum->GetParError(i));
   }
   
}
