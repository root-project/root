/// \notebook -js
/// Customized Morphing function defintion for three parameters 
///
/// \macro_image
/// \macro_output
/// \macro_code
/// \author 04/2016 - Carsten Burgard
#include "RooLagrangianMorphing.h"
#include "RooStringVar.h"
#include "RooArgList.h"
#include "RooFormulaVar.h"
#include "RooRealVar.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TFolder.h"
#include "TLegend.h"
#include "TStyle.h"
using namespace RooFit;

void rf1003_morphcustomdef()
{
  // ---------------------------------------------------------
  // E f f e c t i v e   L a g r a n g i a n   M o r p h i n g
  // =========================================================


  // Define identifier for infilename
  std::string infilename = "input/vbfhwwlvlv_3d.root";

  // Define identifier for input sample foldernames,
  // require 15 samples to describe 3 paramter morphing function
  std::vector<std::string> samplelist = {"kAwwkHwwkSM0","kAwwkHwwkSM1","kAwwkHwwkSM10","kAwwkHwwkSM11","kAwwkHwwkSM12",
                                         "kAwwkHwwkSM13","kAwwkHwwkSM2","kAwwkHwwkSM3","kAwwkHwwkSM4","kAwwkHwwkSM5",
                                         "kAwwkHwwkSM6","kAwwkHwwkSM7","kAwwkHwwkSM8","kAwwkHwwkSM9","kSM0"};

  // Construct list of input samples
  RooArgList inputs;
  for(auto const& sample: samplelist)
  {
     RooStringVar* v = new RooStringVar(sample.c_str(), sample.c_str(), sample.c_str());
     inputs.add(*v);
  }

  // C r e a t e   m o r p h i n g   f u n c t i o n
  // -----------------------------------------------

  // Declare parameters from the Higgs Characterisation Model
  RooRealVar cosa("cosa","cosa",1./sqrt(2));
  RooRealVar lambda("Lambda","Lambda",1000.);
  RooRealVar kSM("kSM","kSM",1.,0.,2.);
  RooRealVar kHww("kHww","kHww",0.,-5.,5.);
  RooRealVar kAww("kAww","kAww",0.,-5.,5.);

  // Declare ArgSets for production and decay couplings
  RooArgSet prodCouplings("vbf");
  RooArgSet decCouplings("hww");

  // Add dependence of paramters to the vertices
  // production
  prodCouplings.add(*(new RooFormulaVar("_gSM"  ,"cosa*kSM",                        RooArgList(cosa,kSM))));
  prodCouplings.add(*(new RooFormulaVar("_gHww" ,"cosa*kHww/Lambda",                RooArgList(cosa,kHww,lambda))));
  prodCouplings.add(*(new RooFormulaVar("_gAww" ,"sqrt(1-(cosa*cosa))*kAww/Lambda", RooArgList(cosa,kAww,lambda))));

  //decay
  decCouplings.add (*(new RooFormulaVar("_gSM"  ,"cosa*kSM",                        RooArgList(cosa,kSM))));
  decCouplings.add (*(new RooFormulaVar("_gHww" ,"cosa*kHww/Lambda",                RooArgList(cosa,kHww,lambda))));
  decCouplings.add (*(new RooFormulaVar("_gAww" ,"sqrt(1-(cosa*cosa))*kAww/Lambda", RooArgList(cosa,kAww,lambda))));

  // Construct customized three parameter morphing functions for the
  // transverse momentum of the di-jet system and the pseudorapidity 
  // of the leading jet in the process VBF Higgs decaying to W+ W-
  // in the Higgs Characterisation Model
  RooLagrangianMorphing::RooLagrangianMorphConfig config;
  config.setCouplings(prodCouplings, decCouplings);
  RooLagrangianMorphing::RooLagrangianMorph morphfunc_ptjj("morphunc_ptjj","morphfuinc_ptjj",infilename.c_str(),"twoSelJets/ptjj", config, inputs);
  RooLagrangianMorphing::RooLagrangianMorph morphfunc_etaj1("morphunc_etaj1","morphfunc_etaj1",infilename.c_str(),"twoSelJets/etaj1", config, inputs);

  // Define identifier for validation sample
  std::string validationsample("v1");
 
  // Set morphing function at parameter configuration of v1
  // available "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9" 
  morphfunc_ptjj.setParameters(validationsample.c_str());
  morphfunc_etaj1.setParameters(validationsample.c_str());

  // Create historgrams from the morphing function at the parameter configurations
  TH1* morph_ptjj_hist = morphfunc_ptjj.createTH1("morphing");
  TH1* morph_etaj1_hist = morphfunc_etaj1.createTH1("morphing");

  // Declare observables 'ptjj' and 'eta_j1'
  RooRealVar ptjj("p^{t}_{jj}", "p^{t}_{jj}", 0, 250);
  RooRealVar etaj1("eta_{j1}", "eta_{j1}", -3, 3);

  // Create a binned dataset that imports the contents of the histogram and associates its contents to the observables
  RooDataHist morph_ptjj_dh("morphed_ptjj", "morphed_ptjj", RooArgList(ptjj), morph_ptjj_hist);
  RooDataHist morph_etaj1_dh("morphed_etaj1", "morphed_etaj1", RooArgList(etaj1), morph_etaj1_hist);

  // P l o t   m o r p h e d   f u n c t i o n   a n d   v a l i d a t i o n   s a m p l e
  // -------------------------------------------------------------------------------------

  // Construct plot frames in 'ptjj' and 'etaj1'
  RooPlot *ptjjframe = ptjj.frame(Title("p^{T}_{jj}"));
  RooPlot *etaj1frame = etaj1.frame(Title("#\\eta_{j1}#"));

  // Read histograms from validation sample folder
  TFile* file = TFile::Open(infilename.c_str(),"READ");
  TFolder* folder = 0;
  file->GetObject(validationsample.c_str(),folder);
  TH1* validation_ptjj_hist = dynamic_cast<TH1*>(folder->FindObject("twoSelJets/ptjj"));
  TH1* validation_etaj1_hist = dynamic_cast<TH1*>(folder->FindObject("twoSelJets/etaj1"));
  validation_ptjj_hist->SetDirectory(NULL);
  validation_ptjj_hist->SetTitle(validationsample.c_str());
  validation_etaj1_hist->SetDirectory(NULL);
  validation_etaj1_hist->SetTitle(validationsample.c_str());
  file->Close();

  // Create binned datasets of the distributions for the validation sample
  RooDataHist validation_ptjj_dh("validation_ptjj_dh", "validation_ptjj_dh", RooArgList(ptjj), validation_ptjj_hist);
  RooDataHist validation_etaj1_dh("validation_etaj1_dh", "validation_etaj1_dh", RooArgList(etaj1), validation_etaj1_hist);

  // Plot morphed and validation distribution on frames
  // pt_jj
  morph_ptjj_dh.plotOn(ptjjframe, Name("morph_ptjj"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
  validation_ptjj_dh.plotOn(ptjjframe, Name("morph_ptjj"));

  // MET
  morph_etaj1_dh.plotOn(etaj1frame, Name("morph_etaj1"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
  validation_etaj1_dh.plotOn(etaj1frame, Name("morph_etaj1"));

   // Draw frames on a canvas
  TCanvas *c = new TCanvas("rf1003_morphcustomdef", "rf1003_morphcustomdef", 800, 400);
  c->Divide(2);
  c->cd(1);
  gPad->SetLeftMargin(0.15);
  ptjjframe->GetYaxis()->SetTitleOffset(1.6);
  ptjjframe->Draw();
  // Create legend description
  TLegend* legend0 = new TLegend(0.60,0.80,0.89,0.89);
  legend0->SetFillColor(kWhite);
  legend0->SetLineColor(kWhite);
  legend0->AddEntry("morph_ptjj","morphing function","L");
  legend0->AddEntry("validation_ptjj","validation sample","PE");
  legend0->Draw();
  c->cd(2);
  gPad->SetLeftMargin(0.15);
  etaj1frame->GetYaxis()->SetTitleOffset(1.6);
  etaj1frame->Draw();
  etaj1frame->SetMaximum(150);
  // Create legend description
  TLegend* legend1 = new TLegend(0.60,0.80,0.89,0.89);
  legend1->SetFillColor(kWhite);
  legend1->SetLineColor(kWhite);
  legend1->AddEntry("morph_etaj1","morphing function","L");
  legend1->AddEntry("validation_etaj1","validation sample","PE");
  legend1->Draw();
  legend1->Draw();
  c->SaveAs("rf1003.pdf");
}
