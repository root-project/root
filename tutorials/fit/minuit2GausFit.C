// @(#)root/minuit2:$Name:  $:$Id: minuit2GausFit.C,v 1.1 2005/10/27 14:11:07 brun Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/


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

  TH1D * h1 = new TH1D("h1","fit histo 1",100, -5, 5. );

  gStyle->SetOptStat(1111111);
  gStyle->SetOptFit(1111111);    

  for (int i = 0; i < n; ++i) { 
     h1->Fill( gRandom->Gaus(0,1) ); 
  }
  
  std::string cname = type + "Canvas" ;    
  std::string ctitle = type + " Gaussian Fit" ;    
  TCanvas *c1 = new TCanvas(cname.c_str(),cname.c_str(),10,10,900,900);
  c1->Divide(2,2);
  
  TH1D * h2 = new TH1D(*h1); 
  TH1D * h3 = new TH1D(*h1); 
  TH1D * h4 = new TH1D(*h1); 
  c1->cd(1);
  cout << "\nDo Fit 1\n";
  h1->Fit("gaus","Q"); 
  h1->SetName("Chi2 Fit");
  h1->Draw();
  c1->cd(2);
  cout << "\nDo Fit 2\n";
  h2->Fit("gaus","VE"); 
  h2->SetName("Chi2 Fit with Minos Erros");
  h2->Draw();
  c1->cd(3);
  cout << "\nDo Fit 3\n";
  h3->Fit("gaus","IE"); 
  h3->SetName("Chi2 Fit with Integral and Minos");
  h3->Draw();
  c1->cd(4);
  cout << "\nDo Fit 4\n";
  h4->Fit("gaus","VLE"); 
  h4->SetName("Likelihood Fit with Minos Erros");
  h4->Draw();

}

void minuit2GausFit() { 

  int n = 1000; 
  testGausFit("Minuit2",n);
  testGausFit("Fumili2",n);

}



