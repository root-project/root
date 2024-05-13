/// \file
/// \ingroup tutorial_fit
/// \notebook -js
/// Perform fits with different configurations using Minuit2
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Lorenzo Moneta

#include "TH1.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TRandom3.h"
#include "TVirtualFitter.h"
#include "TPaveLabel.h"
#include "TStyle.h"

#include <iostream>
#include <string>


void testGausFit( std::string type = "Minuit2", int n = 1000) {

   gRandom = new TRandom3();

   TVirtualFitter::SetDefaultFitter(type.c_str() );

   std::string name;
   name = "h1_" + type;
   TH1D * h1 = new TH1D(name.c_str(),"Chi2 Fit",100, -5, 5. );
   name = "h2_" + type;
   TH1D * h2 = new TH1D(name.c_str(),"Chi2 Fit with Minos Error",100, -5, 5. );
   name = "h3_" + type;
   TH1D * h3 = new TH1D(name.c_str(),"Chi2 Fit with Integral and Minos",100, -5, 5. );
   name = "h4_" + type;
   TH1D * h4 = new TH1D(name.c_str(),"Likelihood Fit with Minos Error",100, -5, 5. );

   gStyle->SetOptStat(1111111);
   gStyle->SetOptFit(1111111);

   for (int i = 0; i < n; ++i) {
      double x = gRandom->Gaus(0,1);
      h1->Fill( x );
      h2->Fill( x );
      h3->Fill( x );
      h4->Fill( x );
   }

   std::string cname = type + "Canvas" ;
   std::string ctitle = type + " Gaussian Fit" ;
   TCanvas *c1 = new TCanvas(cname.c_str(),cname.c_str(),10,10,900,900);
   c1->Divide(2,2);

   c1->cd(1);
   std::cout << "\nDo Fit 1\n";
   h1->Fit("gaus","Q");
   h1->Draw();
   c1->cd(2);
   std::cout << "\nDo Fit 2\n";
   h2->Fit("gaus","E");
   h2->Draw();
   c1->cd(3);
   std::cout << "\nDo Fit 3\n";
   h3->Fit("gaus","IGE");
   h3->Draw();
   c1->cd(4);
   std::cout << "\nDo Fit 4\n";
   h4->Fit("gaus","LE");
   h4->Draw();

}

void minuit2GausFit() {

   int n = 1000;
   testGausFit("Minuit2",n);
   testGausFit("Fumili2",n);

}
