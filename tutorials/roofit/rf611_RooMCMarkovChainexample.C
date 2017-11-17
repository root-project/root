/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
///  'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #611
///
/// Negtive log likelihood minimization with RooMCMarkovChain compared to RooMinuit
/// Plus niceCornerPlot function which can be copied to get nice correlation plots of all parameters
///
/// \macro_image
/// \macro_output
/// \macro_code
/// \author 11/2017 - Oliver Dahme

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooConstVar.h"
#include "RooProdPdf.h"
#include "RooAddPdf.h"
#include "RooMinuit.h"
#include "RooMCMarkovChain.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TLegend.h"
#include "TH1.h"
#include "TH2D.h"
using namespace RooFit ;

TCanvas* niceCornerPlot(RooMCMarkovChain *roomcmc);


void rf611_RooMCMarkovChainexample() {

  //create a canvas
  TCanvas *c1 = new TCanvas("c1","c1",1,1,1920,1080);


    // Observable
  RooRealVar x("x","x",-20,20) ;

  // Model (intentional strong correlations)
  RooRealVar mean_g1("mean1","mean of g2",4.0,-5.0,5.0) ;
  RooRealVar sigma_g1("sigma1","width of g1",1.0,0.0,3) ;
  RooGaussian g1("g1","g1",x,mean_g1,sigma_g1) ;

  RooRealVar mean_g2("mean2","mean of g1",-2.0,-5.0,5.0) ;
  RooRealVar sigma_g2("sigma2","width of g2",1.5,0.0,4.0) ;
  RooGaussian g2("g2","g2",x,mean_g2,sigma_g2) ;

  RooRealVar frac("frac","frac",0.5,0.0,1.0) ;
  RooAddPdf modelmcmc("model","model",RooArgList(g1,g2),frac) ;
  RooAddPdf modelmin("model","model",RooArgList(g1,g2),frac) ;

  // Generate 1000 events
  RooDataSet* datamcmc = modelmcmc.generate(x,1000) ;
  RooDataSet* datamin = modelmin.generate(x,1000) ;

  // Construct unbinned likelihood of model w.r.t. data
  RooAbsReal* nllmcmc = modelmcmc.createNLL(*datamcmc);
  RooAbsReal* nllminuit = modelmin.createNLL(*datamin);

  // Run RooMinuitMCMC fit
  RooMCMarkovChain m(*nllmcmc);
  m.mcmc(2000,150);

  // Run Minuit fit
  RooMinuit mi(*nllminuit);
  mi.migrad();
  mi.hesse();

  // create plot
  RooPlot *frame = x.frame();
  frame->SetTitle("Fit of a double gaus");
  frame->SetXTitle("value");
  datamcmc->plotOn(frame, Name("data"));
  modelmcmc.plotOn(frame, Name("modelmcmc"), LineColor(4));
  modelmin.plotOn(frame, Name("modelmin"), LineColor(2),LineStyle(10));
  frame->Draw();
  TLegend *leg = new TLegend(0.65,0.73,0.86,0.87);
  leg->SetFillColor(kWhite);
  leg->SetLineColor(kWhite);
  leg->AddEntry("data","Data","LP");
  leg->AddEntry("modelmcmc","RooMCMarkovChain fit","L");
  leg->AddEntry("modelmin","Minuit fit","L");
  leg->Draw();

  // create nice corner plot
  TCanvas *corner = niceCornerPlot(&m);

  //Display canvas
  corner->cd();
  c1->cd();

};


/*
saveCornerPlotAs creates a histogram of every parameter with getWalkDisHis and a cornerplot with every pair of parameters. It can be used to see any correlations between the parameters. The Histograms can be used to see graphically if a parameter has an asymertric error or if it has a Gaussian distribution. Picname defines the name of the output file.
*/
TCanvas* niceCornerPlot(RooMCMarkovChain *roomcmc)
{

  gStyle->SetOptStat(0);
  size_t nPar = roomcmc->getNPar();
  int nPads = nPar+ nPar*nPar;
  TCanvas* corner = new TCanvas("corner","corner plot",1,1,nPar*800,nPar*600);
  std::vector<TPad*> pads;
  pads.reserve(nPads);
  for (int i = 0; i < nPar; i++) {
    corner->cd();
    std::string s = std::to_string(i);
    const char* padname = s.c_str();
    TPad *pad = new TPad(padname,padname,0.05,0.02+((nPar-i-1)* 1.0/nPar),0.95,0.97-(i* 1.0/nPar));
    pads.push_back(pad);
    pads[i]->SetFillColor(0);
    pads[i]->Draw();
  }
  for (int i = 0; i < nPar; i++) {
    int subpadindex = 0;
    for (int j = (i+1)*nPar; j < (i+2)*nPar; j++) {
      std::string s = std::to_string(i);
      s = std::to_string(j);
      const char* padname = s.c_str();
      TPad *subpad = new TPad(padname,padname,0.02+(subpadindex* 1.0/nPar),0.05,((subpadindex+1)* 1.0/nPar)-nPar*0.01,0.95,17,3);
      pads[i]->cd();
      pads.push_back(subpad);
      pads[j]->SetFillColor(0);
      pads[j]->Draw();
      subpadindex++;
    }
  }

  std::vector<const char*> names = roomcmc->getNames();

  std::vector<TH1F*> hist1D;
  hist1D.reserve(nPar);
  for (int i = 0; i < nPar;i++) {
    TH1F* hist = roomcmc->getWalkDisHis(names[i],100,kTRUE);
    hist1D.push_back(hist);
  }

  std::vector<TH2D*> hist2D;
  hist2D.reserve(nPar*(nPar-1)/2);
  for (int i = 0; i < nPar; i++) {
    for (int j = i+1; j < nPar; j++) {
      TH2D* hist = roomcmc->getCornerPlot(names[i],names[j],100,100,kTRUE);
      hist2D.push_back(hist);
    }
  }


  size_t Plot1DIndex = 0;
  for (int i = nPar; i < nPads;) {
    pads[i]->cd();
    hist1D[Plot1DIndex]->Draw();
    Plot1DIndex++;
    i+=nPar+1;
  }

  size_t Plot2DIndex = 0;
  for (int i = 2; i < nPar+1; i++) {
    int padindex = i*nPar;
    for (int j = 1; j < i; j++) {
      pads[padindex]->cd();
      hist2D[Plot2DIndex]->SetMarkerStyle(7);
      hist2D[Plot2DIndex]->Draw("colz");
      padindex++;
      Plot2DIndex++;
    }
  }

  // corner->SaveAs(pngname);
  //
  // TFile* file = new TFile("CornerPlot.root", "recreate");
  // file->cd();
  // for (size_t i = 0; i < hist1D.size(); i++) {
  //   hist1D[i]->Write();
  //   delete hist1D[i];
  // }
  // for (size_t i = 0; i < hist2D.size(); i++) {
  //   hist2D[i]->Write();
  //   delete hist2D[i];
  // }
  // file->Close();

  return corner;

  // hist1D.clear();
  // hist2D.clear();


}
