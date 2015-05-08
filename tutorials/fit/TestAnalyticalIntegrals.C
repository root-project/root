//
//  TestAnalyticalIntegrals.C
//  
//
//  Created by Aurélie Flandi on 10.09.14.
//
//

#include <stdio.h>
#include <TStyle.h>
#include <TPaveText.h>
#include <TCanvas.h>
#include <iostream>//for cout
#include <iomanip> //for setprecision
#include <string>
#include <sstream>
#include <TROOT.h>
#include <TChain.h>
#include <TObject.h>
#include <TFile.h>
#include <math.h>
#include "TMath.h"
#include <TF1NormSum.h>
#include <TF1.h>
#include <TH1F.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TProfile.h>
#include <TStopwatch.h>
#include <Math/PdfFuncMathCore.h>//for pdf
#include <Math/ProbFuncMathCore.h>//for cdf


using namespace std;

void TestAnalyticalIntegrals()
{
   //compare analytical integration with numerical one
   
   TF1 *f_pol3 = new TF1("TRY","pol3",-5.,5.);
   f_pol3 -> SetParameters(1.,1.,1.,1.);
   new TCanvas("pol3","pol3",800,1000);
   f_pol3->Draw();
   std::cout<<"ana int pol3 "<<f_pol3->Integral(-5.,5.)<<std::endl;
   
   TF1 *f_cb  = new TF1("MyCrystalBall","ROOT::Math::crystalball_pdf(x,[0],[1],[2],[3])",-5.,5.);
   f_cb -> SetParameters(1,3,3.,0.3);
   new TCanvas("cb","cb",800,1000);
   f_cb->Draw();
   cout<<"num int cb "<<f_cb->Integral(-TMath::Infinity(),TMath::Infinity())<<endl;
   cout<<"ana int cb "<<ROOT::Math::crystalball_cdf(TMath::Infinity(),1,3,3.,0.3)<<endl;
   
   TF1 *f_gaus  = new TF1("MyGauss","gaus",-5.,5.);
   f_gaus -> SetParameters(1.,0.,0.3);
   new TCanvas("gauss","gauss",800,1000);
   f_gaus ->Draw();
   cout<<"num int gauss "<<f_gaus->Integral(-5,5.)<<endl;
   cout<<"ana int gauss "<<ROOT::Math::gaussian_cdf(5.,0.,0.3)-ROOT::Math::gaussian_cdf(-5.,0.,0.3)<<endl;
   
   TF1 *f_exp  = new TF1("MyExp","expo",-5.,5.);
   f_exp -> SetParameters(1.,-0.3);
   new TCanvas("expo","expo",800,1000);
   f_exp ->Draw();
   cout<<"ana int exp "<<f_exp->Integral(-5,5.)<<endl;
  
 }