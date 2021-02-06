/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Fitting the morphing function to pseudodata 
///
/// \macro_image
/// \macro_output
/// \macro_code
/// \author 04/2016 - Carsten Burgard

#include "RooLagrangianMorphing.h"
#include "RooStringVar.h"
#include "RooDataHist.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TFolder.h"
#include "TLegend.h"
#include "TStyle.h"
using namespace RooFit;

void rf1005_morphfit()
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
  // -----------------------------------------------

  // Construct three parameter morphing function for the missing
  // transverse momentum in the process VBF Higgs decaying to  W+ W- in the
  // Higgs Characterisation Model
  RooHCvbfWWMorph morphfunc("morphfunc", "morphfunc", infilename.c_str(), "twoSelJets/pth_shower", inputs);

  // Define identifier for validation sample
  std::string validationsample = "v3";

  // Set morphing function at parameter configuration of v3
  // available "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9" 
  morphfunc.setParameters(validationsample.c_str());

  // Create histograms from the morphing function at the parameter configuration
  TH1* morph_hist = morphfunc.createTH1("morphing");

  // Declare observable 'pth'
  RooRealVar* pth = morphfunc.getObservable();

  // F i t t i n g   t o   m o r p h i n g   f u n c t i o n   a n d   v i s u a l i s a t i o n
  // -------------------------------------------------------------------------------------------

  // Construct plot frames in 'dphijj' and 'met'
  RooPlot *pthframe = pth->frame(Title("p_{T}^{H}"));

  // Read histograms from validation sample folder
  TFile* file = TFile::Open(infilename.c_str(),"READ");
  TFolder* folder = 0;
  file->GetObject(validationsample.c_str(),folder);
  TH1* validation_hist = dynamic_cast<TH1*>(folder->FindObject("twoSelJets/pth_shower"));
  validation_hist->SetDirectory(NULL);
  validation_hist->SetTitle(validationsample.c_str());

  // S e t u p   t h e   f i t 
  // -------------------------

  // Convert the pseudodata ot a RooDataHist 
  RooDataHist* pseudodata_dh = RooLagrangianMorphing::makeDataHistogram(validation_hist, pth, "validation");

  // Configure parameters for the fit
  morphfunc.setParameters(validationsample.c_str());
  morphfunc.setParameterConstant("Lambda",true);
  morphfunc.setParameterConstant("cosa",true);

  // Randomize the parameters by 2 standard deviations to give the fit something to do
  morphfunc.randomizeParameters(3);
  morphfunc.printParameters();

  // Run the fit
  auto x = morphfunc.getPdf()->fitTo(*pseudodata_dh, RooFit::SumW2Error(true), RooFit::Optimize(false));
  morphfunc.printParameters();

  // Plot fiited and pseudodata distribution on frame
  TH1* fitresult_hist = morphfunc.createTH1("fit result");
  RooDataHist* fitresult_dh = RooLagrangianMorphing::makeDataHistogram(fitresult_hist, pth, "fitresult");

  fitresult_dh->plotOn(pthframe, Name("morphfit"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
  pseudodata_dh->plotOn(pthframe, Name("pseudodata"));

  // Draw frame on a canvas
  TCanvas* c =  new TCanvas("rf1005_morphfit", "rf1005_morphfit", 400, 400);
  gPad->SetLeftMargin(0.15);
  pthframe->GetYaxis()->SetTitleOffset(1.6);
  pthframe->Draw();
}
