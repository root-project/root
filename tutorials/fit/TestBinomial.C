// Example of performing a fit to a set of data which have binomial errors like those 
// derived from the division of two histograms. 
// Thre different fits are performed and compared:  
//
//   -  simple least square fit to the divided histogram obtained from TH1::Divide with option b
//   -  least square fit to the TGraphAsymmErrors obtained from TGraphAsymmErrors::BayesDivide
//   -  likelihood fit performed on the dividing histograms using binomial statistics with the TBinomialEfficiency class
// 
// The first two methods are biased while the last one  is statistical correct. 
// Running the script passing an integer value n larger than 1, n fits are performed and the bias are also shown.   
// To run the script : 
// 
//  root[0]: .x TestBinomial.C+
// 
//           .x TestBinomial.C+(100)  to show the bias performing 100 fits
// 
//
#include "TBinomialEfficiencyFitter.h"
#include "TGraphAsymmErrors.h"
#include "TVirtualFitter.h"
#include "TH1.h"
#include "TRandom3.h"
#include "TF1.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLegend.h"
#include <cassert>
#include <iostream>

void TestBinomial(int nloop=1)
{

   TH1D *hfirst = new TH1D("hfirst", "hfirst", 100, -5, 5);
   TH1D *hsecond = new TH1D("hsecond", "hsecond", 100, -5, 5);
   TH1D *hresult = (TH1D*)hsecond->Clone();
   hfirst->Sumw2();
   hsecond->Sumw2();

   TH1D *h0 = new TH1D("h0","Bias  Histogram fit",100,-5,5);
   TH1D *h1 = new TH1D("h1","Bias Graph fit",100,-5,5);
   TH1D *h2 = new TH1D("h2","Bias Binomial fit",100,-5,5);

   //TVirtualFitter::SetDefaultFitter("Minuit2");


   TGraphAsymmErrors * gr = new TGraphAsymmErrors();

   TF1 * f0 = 0;
   TF1 * f1 = 0;
   TF1 * f2 = new TF1("f2","pol0", -5, 5);

   gStyle->SetOptFit(1111);

   double  p = 0.9;


   for (int iloop = 0; iloop < nloop; ++iloop) { // generate many times to see the bias


      hfirst->Reset();
      hsecond->Reset();
      Double_t x, xu;
      TRandom3 r(0);
      for (Int_t i=0; i<1000; i++) {
         x = r.Gaus();
         hfirst->Fill(x);
         xu = r.Uniform(0, 1);
         if (xu<=p)
            hsecond->Fill(x);
      }

      hresult->Divide(hsecond, hfirst, 1, 1, "b");

      // use TGraphAsymmErrors
      gr->BayesDivide(hsecond,hfirst,"w");

      hresult->Fit("pol0", "0");
      f0 = hresult->GetFunction("pol0");
      assert(f0 != 0);

      gr->Fit("pol0");
      f1 = gr->GetFunction("pol0");
      assert(f0 != 0);
   

      f2->SetParameter(0, .5);

      TBinomialEfficiencyFitter bef(hsecond, hfirst);
      Int_t status = bef.Fit(f2,"r");
      if (status!=0) { 
         std::cerr << "Error performing binomial efficiency fit " << std::endl;
         continue; 
      }

      double p0 = f0->GetParameter(0);
      double e0 = f0->GetParError(0);
      double p1 = f1->GetParameter(0);
      double e1 = f1->GetParError(0);
      double p2 = f2->GetParameter(0);
      double e2 = f2->GetParError(0);
      h0->Fill( (p0-p)/e0 );
      h1->Fill( (p1-p)/e1 );
      h2->Fill( (p2-p)/e2 );


   }
   

   TCanvas * c1;  
   if (nloop > 1) {   
      c1 = new TCanvas("c1","Binomial Fit",10,10,1000,800); 
      c1->Divide(2,2);
   }
   else 
      c1 = new TCanvas("c1","Binomial Fit");
  
   c1->cd(1);
   gr->SetTitle("Efficiency histogram");
   gr->Draw("AP");
   f0->Draw("SAME");
   f1->SetLineColor(kBlue);
   f1->Draw("same");
   f2->SetLineColor(kRed);
   f2->Draw("same");

   TLegend * l = new TLegend(0.78,0.1,0.97,0.3);
   l->AddEntry(f0, "Histogram fit");
   l->AddEntry(f1, "Graph     fit");
   l->AddEntry(f2, "Binomial  fit");
   l->Draw();

   if (nloop > 1) { 
      h0->GetXaxis()->SetTitle("Delta/Sigma");
      h1->GetXaxis()->SetTitle("Delta/Sigma");
      h2->GetXaxis()->SetTitle("Delta/Sigma");
      c1->cd(2);
      h0->Draw();
      c1->cd(3);
      h1->Draw();
      c1->cd(4);
      h2->Draw();
   }


   printf("Histo eff fit results\n");
   printf("%f +- %f \tDelta=%f  sigma\n", f0->GetParameter(0), f0->GetParError(0),(f0->GetParameter(0)-p)/f0->GetParError(0));

   printf("Graph eff fit results\n");
   printf("%f +- %f \tDelta=%f  sigma\n", f1->GetParameter(0), f1->GetParError(0),(f1->GetParameter(0)-p)/f1->GetParError(0));

   printf("bin eff fit results\n");
   printf("%f +- %f \tDelta=%f  sigma\n", f2->GetParameter(0), f2->GetParError(0),(f2->GetParameter(0)-p)/f2->GetParError(0));


   printf("\n");
}
