/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Morphing function for two parameters 
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
#include "RooPlot.h"
using namespace RooFit;

void rf1001_morph2params()
{
   // ---------------------------------------------------------
   // E f f e c t i v e   L a g r a n g i a n   M o r p h i n g
   // =========================================================


   // C r e a t e   m o r p h i n g   f u n c t i o n
   // ------------------------------------------------
  
   // Define identifier for infilename
   std::string infilename = "input/ggfhzz4l_2d.root";
   // Define identifier for input sample foldernames,
   // require three samples to describe two parameter morphing funciton
   std::vector<std::string> samplelist = {"s1", "s2", "s3"};
   
   // Construct list of input samples
   RooArgList inputs;
   for(auto const& sample: samplelist)
   {
      RooStringVar* v = new RooStringVar(sample.c_str(), sample.c_str(), sample.c_str());
      inputs.add(*v);
   }

   // Construct two parameter morphing functions for pseudorapidity and
   // the transverse momentum of the Higgs in the process ggF Higgs
   // decaying to ZZ in the Higgs Characterisation Model
   RooHCggfZZMorphFunc morphfunc_eta("ggfHZZ", "ggfHZZ", infilename.c_str(), "base/etah", inputs);
   RooHCggfZZMorphFunc morphfunc_pt("ggfHZZ", "ggfHZZ", infilename.c_str(), "base/pth", inputs);

   // Set morphing function at parameter configuration of v1
   // available "v0", "v1"
   std::string validationsample = "v1";
   morphfunc_eta.setParameters(validationsample.c_str());
   morphfunc_pt.setParameters(validationsample.c_str());

   // Create histograms from the morphing function at the parameter configuration
   TH1* morph_eta_hist = morphfunc_eta.createTH1("morphing");
   TH1* morph_pt_hist  = morphfunc_pt.createTH1("morphing");

   // Declare observables 'phi' & 'pt'
   RooRealVar eta("eta_H", "eta_H", -3.1415, 3.1415);
   RooRealVar pt("pt_H", "pT_H", 0, 250);

   // Create a binned dataset that imports the contents of the histogram and associates its contents to observable
   RooDataHist morph_eta_dh("morphed_eta", "morphed_eta", RooArgList(eta), morph_eta_hist);
   RooDataHist morph_pt_dh("morphed_pt", "morphed_pt", RooArgList(pt), morph_pt_hist);
  
   // P l o t   m o r p h e d   f u n c t i o n   a n d   v a l i d a t i o n   s a m p l e
   // -------------------------------------------------------------------------------------

   // Construct plot frames in 'phi' and 'pt'
   RooPlot *etaframe = eta.frame(Title("#\\eta^H#"));
   RooPlot *ptframe  = pt.frame(Title("p_{t}^{H}"));
 
   // Read histograms from validation sample folder
   TFile* file = TFile::Open(infilename.c_str(), "READ");
   TFolder* folder = 0;
   file->GetObject(validationsample.c_str(), folder);
   TH1* validation_eta_hist = dynamic_cast<TH1*>(folder->FindObject("base/etah"));
   TH1* validation_pt_hist  = dynamic_cast<TH1*>(folder->FindObject("base/pth"));
   validation_eta_hist->SetDirectory(NULL);
   validation_eta_hist->SetTitle(validationsample.c_str());
   validation_pt_hist->SetDirectory(NULL);
   validation_pt_hist->SetTitle(validationsample.c_str());
   file->Close();

   // Create binned datasets of the distributions for the validation sample
   RooDataHist validation_eta_dh("validation_eta_dh", "validation_eta_dh", RooArgList(eta), validation_eta_hist); 
   RooDataHist validation_pt_dh("validation_pt_dh", "validation_pt_dh", RooArgList(pt), validation_pt_hist); 

   // Plot morphed and validation distribution on frames
   //phi_H
   morph_eta_dh.plotOn(etaframe, Name("morph_eta"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
   validation_eta_dh.plotOn(etaframe, Name("validation_eta"));

   //pt_H
   morph_pt_dh.plotOn(ptframe, Name("morph_pt"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
   validation_pt_dh.plotOn(ptframe, Name("validation_pt"));

   // Draw frame on a canvas
   TCanvas *c = new TCanvas("rf1001_morph2params", "rf1001_morph2params", 800, 400);
   c->Divide(2);
   c->cd(1);
   gPad->SetLeftMargin(0.15);
   etaframe->GetYaxis()->SetTitleOffset(1.6);
   etaframe->SetMinimum(0.1);
   etaframe->SetMaximum(1);
   etaframe->Draw();
   // Create legend description
   TLegend* legend0 = new TLegend(0.60,0.80,0.89,0.89);
   legend0->SetFillColor(kWhite);
   legend0->SetLineColor(kWhite);
   legend0->AddEntry("morph_eta","morphing function","L");
   legend0->AddEntry("validation_eta","validation sample","PE");
   legend0->Draw();
   c->cd(2);
   gPad->SetLeftMargin(0.15);
   ptframe->GetYaxis()->SetTitleOffset(1.6);
   ptframe->SetMinimum(0.1);
   ptframe->SetMaximum(1);
   ptframe->Draw();
   // Create legend description
   TLegend* legend1 = new TLegend(0.60,0.80,0.89,0.89);
   legend1->SetFillColor(kWhite);
   legend1->SetLineColor(kWhite);
   legend1->AddEntry("morph_pt","morphing function","L");
   legend1->AddEntry("validation_pt","validation sample","PE");
   legend1->Draw();
   legend1->Draw();
}
