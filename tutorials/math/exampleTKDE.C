/// \file
/// \ingroup tutorial_math
/// \notebook
/// Example of using the TKDE class (kernel density estimator)
///
/// \macro_image
/// \macro_code
///
/// \authors Lorenzo Moneta, Bartolomeu Rabacal (Dec 2010)

#include "TH1.h"
#include "TF1.h"
#include "TKDE.h"
#include "TCanvas.h"
/*#include "TStopwatch.h"*/
#include "TRandom.h"
#include "Math/DistFunc.h"
#include "TLegend.h"

// test TKDE

void exampleTKDE(int n = 1000) {

   // generate some gaussian points

   int nbin = 100;
   double xmin = 0;
   double xmax = 10;

   TH1D * h1 = new TH1D("h1","h1",nbin,xmin,xmax);

   // generate some points with bi- gaussian distribution

   std::vector<double> data(n);
   for (int i = 0; i < n; ++i) {
      if (i < 0.4*n) {
         data[i] = gRandom->Gaus(2,1);
         h1->Fill(data[i]);
      }
      else {
         data[i] = gRandom->Gaus(7,1.5);
         h1->Fill(data[i]);
      }
   }

   // scale histogram
   h1->Scale(1./h1->Integral(),"width" );
   h1->SetStats(false);
   h1->SetTitle("Bi-Gaussian");
   h1->Draw();

   // drawn true normalized density
   TF1 * f1 = new TF1("f1","0.4*ROOT::Math::normal_pdf(x,1,2)+0.6*ROOT::Math::normal_pdf(x,1.5,7)",xmin,xmax);
   f1->SetLineColor(kGreen+2);
   f1->Draw("SAME");

   // create TKDE class
   double rho = 1.0; //default value
   TKDE * kde = new TKDE(n, &data[0], xmin,xmax, "", rho);
   //kde->Draw("ConfidenceInterval@0.95 Same");
   kde->Draw("SAME");

   TLegend * legend = new TLegend(0.6,0.7,0.9,0.95);
   legend->AddEntry(f1,"True function");
   legend->AddEntry(kde->GetDrawnFunction(),"TKDE");
   legend->AddEntry(kde->GetDrawnLowerFunction(),"TKDE - #sigma");
   legend->AddEntry(kde->GetDrawnUpperFunction(),"TKDE + #sigma");
   legend->Draw();
}
