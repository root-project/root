#include <stdio.h>
#include <TMath.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TChain.h>
#include <TObject.h>
#include <TRandom.h>
#include <TF1NormSum.h>
#include <TF1.h>
#include <TH1F.h>
#include <Math/PdfFuncMathCore.h>  //for the crystalball function


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

   Int_t NEvents = 1e6;
   Int_t NBins   = 1e3;
   
   TF1 *f_cb    = new TF1("MyCrystalBall","ROOT::Math::crystalball_pdf(x,[0],[1],[2],[3])",-5.,5.);
   TF1 *f_exp   = new TF1("MyExponential","expo",-5.,5.);
   
   // I.:
  // f_exp-> SetParameters(0.,-0.3);
   //f_cb -> SetParameters(1,2,3,0.3);
   
   // CONSTRUCTION OF THE TF1NORMSUM OBJECT ........................................
   // 1) :
   //TF1NormSum *fnorm_exp_cb = new TF1NormSum("0.2*expo + MyCrystalBall",-5.,5.);
   // 2) :
   //TF1NormSum *fnorm_exp_cb = new TF1NormSum("0.2*expo + MyCrystalBall");
   // 3) :
   TF1NormSum *fnorm_exp_cb = new TF1NormSum(f_exp,f_cb,0.2,1);
   // 4) :
   //const std::vector < TF1*     > functions  = {f_exp, f_cb};
   //const std::vector < Double_t > coeffs     = {0.2,1};
   //TF1NormSum *fnorm_exp_cb = new TF1NormSum(functions,coeffs);

   // II. :
   //fnorm_exp_cb -> SetParameters(1.,1.,-0.3,1,2,3,0.3);
   
   TF1   * f_sum = new TF1("fsum", *fnorm_exp_cb, -5., 5., fnorm_exp_cb->GetNpar());
   f_sum->Draw();
   
   // III.:
   f_sum -> SetParameters(2e5, 8e5, -0.3, 1., 2., 3, 0.3);
  
   //HISTOGRAM TO FIT ..............................................................
   TH1F *h_sum = new TH1F("h_ExpCB", "Exponential Bkg + CrystalBall function", NBins, -5., 5.);
   for (int i=0; i<NEvents; i++)
   {
      h_sum -> Fill(f_sum -> GetRandom());
   }
   h_sum -> Sumw2();
   h_sum -> Scale(1., "width");
   h_sum->Draw();
   TH1F *h_copy = new TH1F(*h_sum);

   //fit
   new TCanvas("Fit","Fit",800,1000);
   h_copy -> Fit("fsum");
   h_copy -> Draw();
   
   
   
}
