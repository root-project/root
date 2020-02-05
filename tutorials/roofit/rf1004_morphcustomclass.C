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

// C u s t o m   m o r p h i n g   c l a s s
// -----------------------------------------

// Declare custom morphing class
class RooHCCustomMorphFunc : public RooLagrangianMorphFunc
{
  MAKE_ROOLAGRANGIANMORPH(Func,RooHCCustomMorphFunc)
  ClassDef(RooHCCustomMorphFunc,1)
  protected:
  void makeCouplings()
  {
    // Declare ArgSet for parameters relevant to the process
    RooArgSet kappas("vbfWW");

    // Declare parameters from the Higgs Characterisation Model
    RooRealVar* cosa = new RooRealVar("cosa","cosa",1./sqrt(2));
    RooRealVar* lambda = new RooRealVar("Lambda","Lambda",1000.);
    RooRealVar* kSM = new RooRealVar("kSM","kSM",1.,0.,2.);
    RooRealVar* kHww = new RooRealVar("kHww","kHww",0.,-1.,1.);
    RooRealVar* kAww = new RooRealVar("kAww","kAww",0.,-1.,1.);

    // Add parameters to the parameter set
    kappas.add(*cosa);
    kappas.add(*lambda);
    kappas.add(*kSM);
    kappas.add(*kHww);
    kappas.add(*kAww);

    // Declare ArgSets for production and decay couplings
    RooArgSet prodCouplings("vbf");
    RooArgSet decCouplings("hww");

    // Add dependence of parameters to the vertices
    // production
    prodCouplings.add(*(new RooFormulaVar("_gSM"  ,"cosa*kSM",                        RooArgList(*cosa,*kSM))));
    prodCouplings.add(*(new RooFormulaVar("_gHww" ,"cosa*kHww/Lambda",                RooArgList(*cosa,*kHww,*lambda))));
    prodCouplings.add(*(new RooFormulaVar("_gAww" ,"sqrt(1-(cosa*cosa))*kAww/Lambda", RooArgList(*cosa,*kAww,*lambda))));

    // decay
    decCouplings.add (*(new RooFormulaVar("_gSM"  ,"cosa*kSM",                        RooArgList(*cosa,*kSM))));
    decCouplings.add (*(new RooFormulaVar("_gHww" ,"cosa*kHww/Lambda",                RooArgList(*cosa,*kHww,*lambda))));
    decCouplings.add (*(new RooFormulaVar("_gAww" ,"sqrt(1-(cosa*cosa))*kAww/Lambda", RooArgList(*cosa,*kAww,*lambda))));
    this->setCouplings(prodCouplings, decCouplings);
    // Setup morphing function
    this->setup();
  }
};

ClassImp(RooHCCustomMorphFunc);

void rf1004_morphcustomclass()
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

  // Construct thee parameter customized morphing functions for the invariant mass
  // of the di-jet system and the number of reconstructed jets in the
  // process VBF Higgs decaying to W+ W- in the Higgs Characterisation Model
  RooHCCustomMorphFunc morphfunc_mjj("morphunc_mjj", "morphfuinc_mjj", infilename.c_str(), "twoSelJets/mjj", inputs);
  RooHCCustomMorphFunc morphfunc_nj("nj", "morphfunc_nj", infilename.c_str(), "twoSelJets/nj", inputs);

  // Define identifier for validation sample
  std::string validationsample("v1");
 
  // Set morphing function at parameter configuration of v1
  // available "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9" 
  morphfunc_mjj.setParameters(validationsample.c_str());
  morphfunc_nj.setParameters(validationsample.c_str());

  // Create historgrams from the morphing function at the parameter configurations
  TH1* morph_mjj_hist = morphfunc_mjj.createTH1("morphing");
  TH1* morph_nj_hist = morphfunc_nj.createTH1("morphing");

  // Declare observables 'mjj' and 'nj'
  RooRealVar mjj("m_{jj}", "m_{jj}", 0, 1000);
  RooRealVar nj("n_{j}", "n_{j}", 0, 6);

  // Create a binned dataset that imports the contents of the histogram and associates its contents to the observables
  RooDataHist morph_mjj_dh("morphed_mjj", "morphed_mjj", RooArgList(mjj), morph_mjj_hist);
  RooDataHist morph_nj_dh("morphed_nj", "morphed_nj", RooArgList(nj), morph_nj_hist);

  // P l o t   m o r p h e d   f u n c t i o n   a n d   v a l i d a t i o n   s a m p l e
  // -------------------------------------------------------------------------------------

  // Construct plot frames in 'mjj' and 'nj'
  RooPlot *mjjframe = mjj.frame(Title("m_{jj}"));
  RooPlot *njframe = nj.frame(Title("n_j"));

  // Read histograms from validation sample folder
  TFile* file = TFile::Open(infilename.c_str(),"READ");
  TFolder* folder = 0;
  file->GetObject(validationsample.c_str(),folder);
  TH1* validation_mjj_hist = dynamic_cast<TH1*>(folder->FindObject("twoSelJets/mjj"));
  TH1* validation_nj_hist = dynamic_cast<TH1*>(folder->FindObject("twoSelJets/nj"));
  validation_mjj_hist->SetDirectory(NULL);
  validation_mjj_hist->SetTitle(validationsample.c_str());
  validation_nj_hist->SetDirectory(NULL);
  validation_nj_hist->SetTitle(validationsample.c_str());
  file->Close();

  // Create binned datasets of the distributions for the validation sample
  RooDataHist validation_mjj_dh("validation_mjj_dh", "validation_mjj_dh", RooArgList(mjj), validation_mjj_hist);
  RooDataHist validation_nj_dh("validation_nj_dh", "validation_nj_dh", RooArgList(nj), validation_nj_hist);

  // Plot morphed and validation distribution on frames
  // mjj
  morph_mjj_dh.plotOn(mjjframe, Name("morph_mjj"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
  validation_mjj_dh.plotOn(mjjframe, Name("morph_mjj"));

  // MET
  morph_nj_dh.plotOn(njframe, Name("morph_nj"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
  validation_nj_dh.plotOn(njframe, Name("morph_nj"));

  // Draw frames on a canvas
  TCanvas *c = new TCanvas("rf1004_morphcustomclass", "rf1004_morphcustomclass", 800, 400);
  c->Divide(2);
  c->cd(1);
  gPad->SetLeftMargin(0.15);
  mjjframe->GetYaxis()->SetTitleOffset(1.6);
  mjjframe->Draw();
  // Create legend description
  TLegend* legend0 = new TLegend(0.60,0.80,0.89,0.89);
  legend0->SetFillColor(kWhite);
  legend0->SetLineColor(kWhite);
  legend0->AddEntry("morph_mjj","morphing function","L");
  legend0->AddEntry("validation_mjj","validation sample","PE");
  legend0->Draw();
  c->cd(2);
  gPad->SetLeftMargin(0.15);
  njframe->GetYaxis()->SetTitleOffset(1.6);
  njframe->Draw();
  // Create legend description
  TLegend* legend1 = new TLegend(0.60,0.80,0.89,0.89);
  legend1->SetFillColor(kWhite);
  legend1->SetLineColor(kWhite);
  legend1->AddEntry("morph_nj","morphing function","L");
  legend1->AddEntry("validation_nj","validation sample","PE");
  legend1->Draw();
  legend1->Draw();
}
