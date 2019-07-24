/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Morphing function for three parameters 
///
/// \macro_image
/// \macro_output
/// \macro_code
/// \author 04/2016 - Carsten Burgard
#include "RooLagrangianMorphing.h"
#include "RooStringVar.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TFolder.h"
#include "TLegend.h"
#include "TStyle.h"
using namespace RooFit;

void rf1002_morph3params()
{
  // ---------------------------------------------------------
  // E f f e c t i v e   L a g r a n g i a n   M o r p h i n g
  // =========================================================

  // Define identifier for infilename
  std::string infilename = "input/vbfhwwlvlv_3d.root";

  // Define identifier for input sample foldernames,
  // require 15 samples to describe three paramter morphing function
  std::vector<std::string> samplelist = {"kAwwkHwwkSM0","kAwwkHwwkSM1","kAwwkHwwkSM10","","kAwwkHwwkSM11","kAwwkHwwkSM12",
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
  // ------------------------------------------------
  
  // Construct three parameter morphing functions for opening angle of the
  // final-state jets and the missing transverse momentum in the process VBF
  //  Higgs decaying to W+ W- in the Higgs Characterisation Model
  RooHCvbfWWMorphFunc morphfunc_dphijj("morphfunc_dphijj", "morphfunc_dphijj", infilename.c_str(), "twoSelJets/dphijj", inputs);
  RooHCvbfWWMorphFunc morphfunc_met("morphfunc_met", "morphfunc_met", infilename.c_str(), "twoSelJets/MET", inputs);

  // Define identifier for validation sample
  std::string validationsample = "v1";

  // Set morphing function at parameter configuration of v1
  // available "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9" 
  morphfunc_dphijj.setParameters(validationsample.c_str());
  morphfunc_met.setParameters(validationsample.c_str());

  // Create histograms from the morphing function at the parameter configuration
  TH1* morph_dphijj_hist = morphfunc_dphijj.createTH1("morphing");
  TH1* morph_met_hist = morphfunc_met.createTH1("morphing");

  // Declare observables 'dphijj' and 'met'
  RooRealVar dphijj("dphi_jj", "dphi_jj", -3.1415, 3.1415);
  RooRealVar met("met", "met", 0, 250);

  // Create a binned dataset that imports the contents of the histogram and associates its contents to observable
  RooDataHist morph_dphijj_dh("morphed_dphijj", "morphed_dphijj", RooArgList(dphijj), morph_dphijj_hist);
  RooDataHist morph_met_dh("morphed_met", "morphed_met", RooArgList(met), morph_met_hist);   

  // P l o t   m o r p h e d   f u n c t i o n   a n d   v a l i d a t i o n   s a m p l e
  // -------------------------------------------------------------------------------------

  // Construct plot frames in 'dphijj' and 'met'
  RooPlot *dphijjframe = dphijj.frame(Title("#\\Delta\\phi_{jj}#"));
  RooPlot *metframe = met.frame(Title("MET"));

  // Read histograms from validation sample folder
  TFile* file = TFile::Open(infilename.c_str(),"READ");
  TFolder* folder = 0;
  file->GetObject(validationsample.c_str(),folder);
  TH1* validation_dphijj_hist = dynamic_cast<TH1*>(folder->FindObject("twoSelJets/dphijj"));
  TH1* validation_met_hist = dynamic_cast<TH1*>(folder->FindObject("twoSelJets/MET"));
  validation_dphijj_hist->SetDirectory(NULL);
  validation_dphijj_hist->SetTitle(validationsample.c_str());
  validation_met_hist->SetDirectory(NULL);
  validation_met_hist->SetTitle(validationsample.c_str());
  file->Close();

  // Create binned datasets of the distributions for the validation sample
  RooDataHist validation_dphijj_dh("validation_dphijj_dh", "validation_dphijj_dh", RooArgList(dphijj), validation_dphijj_hist);
  RooDataHist validation_met_dh("validation_met_dh", "validation_met_dh", RooArgList(met), validation_met_hist);

  // Plot morphed and validation distribution on frames
  // dphi_jj
  morph_dphijj_dh.plotOn(dphijjframe, Name("morph_dphijj"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
  validation_dphijj_dh.plotOn(dphijjframe, Name("morph_dphijj"));
  
  // MET
  morph_met_dh.plotOn(metframe, Name("morph_met"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
  validation_met_dh.plotOn(metframe, Name("morph_met"));

   // Draw frames on a canvas
   TCanvas *c = new TCanvas("rf1002_morph3params", "rf1001_morph3params", 800, 400);
   c->Divide(2);
   c->cd(1);
   gPad->SetLeftMargin(0.15);
   dphijjframe->GetYaxis()->SetTitleOffset(1.6);
   dphijjframe->Draw();
   // Create legend description
   TLegend* legend0 = new TLegend(0.60,0.80,0.89,0.89);
   legend0->SetFillColor(kWhite);
   legend0->SetLineColor(kWhite);
   legend0->AddEntry("morph_dphijj","morphing function","L");
   legend0->AddEntry("validation_dphijj","validation sample","PE");
   legend0->Draw();
   c->cd(2);
   gPad->SetLeftMargin(0.15);
   metframe->GetYaxis()->SetTitleOffset(1.6);
   metframe->Draw();
   // Create legend description
   TLegend* legend1 = new TLegend(0.60,0.80,0.89,0.89);
   legend1->SetFillColor(kWhite);
   legend1->SetLineColor(kWhite);
   legend1->AddEntry("morph_met","morphing function","L");
   legend1->AddEntry("validation_met","validation sample","PE");
   legend1->Draw();
   legend1->Draw();
}
