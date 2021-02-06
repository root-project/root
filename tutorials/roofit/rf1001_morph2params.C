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
#include "Rtypes.h"
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
   RooHCggfZZMorph morphfunc_eta("ggfHZZ", "ggfHZZ", infilename.c_str(), "base/etah", inputs);
   RooHCggfZZMorph morphfunc_pt("ggfHZZ", "ggfHZZ", infilename.c_str(), "base/pth", inputs);
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
  
   // P l o t   m o r p h e d   f u n c t i o n , i n p u t   s a m p l e s   a n d   v a l i d a t i o n   s a m p l e
   // -------------------------------------------------------------------------------------

   // Construct plot frames in 'phi' and 'pt'
   RooPlot *etaframe_input = eta.frame(Title("\\eta^{H} of input samples"));
   RooPlot *ptframe_input  = pt.frame(Title("p_{t}^{H} of input samples"));
   RooPlot *etaframe_morph = eta.frame(Title("\\eta^{H} of validation sample"));
   RooPlot *ptframe_morph  = pt.frame(Title("p_{t}^{H} of validation sample"));

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

   // Create binned datasets of the distributions for the validation sample
   RooDataHist validation_eta_dh("validation_eta_dh", "validation_eta_dh", RooArgList(eta), validation_eta_hist); 
   RooDataHist validation_pt_dh("validation_pt_dh", "validation_pt_dh", RooArgList(pt), validation_pt_hist); 

   // Read histograms from input sample folders
   std::vector<TH1*>  samples_pt_hist, samples_eta_hist;
   std::vector<RooDataHist> samples_pt_dh, samples_eta_dh;
   for(auto sample : samplelist) {
     file->GetObject(sample.c_str(), folder);
     TH1* sample_eta_hist = dynamic_cast<TH1*>(folder->FindObject("base/etah"));
     TH1* sample_pt_hist  = dynamic_cast<TH1*>(folder->FindObject("base/pth"));
     sample_eta_hist->SetDirectory(NULL);
     sample_eta_hist->SetTitle(sample.c_str());
     sample_pt_hist->SetDirectory(NULL);
     sample_pt_hist->SetTitle(sample.c_str());
     samples_eta_hist.push_back(sample_eta_hist);
     samples_pt_hist.push_back(sample_pt_hist);
     // Create binned dataset of the distributions for the input sample
     RooDataHist sample_eta_dh(sample.c_str(), sample.c_str(), RooArgList(eta), sample_eta_hist); 
     RooDataHist sample_pt_dh(sample.c_str(), sample.c_str(), RooArgList(pt), sample_pt_hist);
     samples_eta_dh.push_back(sample_eta_dh);
     samples_pt_dh.push_back(sample_pt_dh);
   }
   file->Close();

   // Plot input distribution on frames
   //phi_H
   std::vector<Color_t> colors = {kRed, kGray, kOrange};
   int i_color = 0;

   for(auto it_eta = samples_eta_dh.begin(); it_eta != samples_eta_dh.end(); ++it_eta){
     RooDataHist sample_eta_dh = *it_eta;
     sample_eta_dh.plotOn(etaframe_input, Name(sample_eta_dh.GetName()), MarkerSize(0.5), MarkerColor(colors[i_color]));
     ++i_color;
   }

   i_color = 0;

   for(auto it_pt = samples_pt_dh.begin(); it_pt != samples_pt_dh.end(); ++it_pt){
     RooDataHist sample_pt_dh = *it_pt;
     sample_pt_dh.plotOn(ptframe_input, Name(sample_pt_dh.GetName()), MarkerSize(0.5), MarkerColor(colors[i_color]));
     ++i_color;
   }
   // Plot morphed and validation distribution on frames
   //phi_H
   morph_eta_dh.plotOn(etaframe_morph, Name("morph_eta"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
   validation_eta_dh.plotOn(etaframe_morph, Name("validation_eta"), MarkerSize(0.5));

   //pt_H
   morph_pt_dh.plotOn(ptframe_morph, Name("morph_pt"), DrawOption("E3"), LineStyle(kSolid), LineColor(kBlue), FillColor(kBlue));
   validation_pt_dh.plotOn(ptframe_morph, Name("validation_pt"), MarkerSize(0.5));

   // Draw frame on a canvas
   TCanvas *c = new TCanvas("rf1001_morph2params", "rf1001_morph2params", 800, 800);
   c->Divide(2,2);
   c->cd(1);
   gPad->SetLeftMargin(0.15);
   etaframe_input->GetYaxis()->SetTitleOffset(1.6);
   etaframe_input->SetMinimum(0.1);
   etaframe_input->SetMaximum(1);
   etaframe_input->Draw();

   // Create legend description
   TLegend legend1(0.80,0.75,0.89,0.89);
   legend1.SetFillColor(kWhite);
   legend1.SetLineColor(kWhite); 
   for(auto it_eta = samples_eta_dh.begin(); it_eta != samples_eta_dh.end(); ++it_eta){
     RooDataHist sample_eta_dh = *it_eta;
     legend1.AddEntry(sample_eta_dh.GetTitle(), sample_eta_dh.GetTitle(), "PE");
   }   
   legend1.Draw();
  
   c->cd(2);
   gPad->SetLeftMargin(0.15);
   ptframe_input->GetYaxis()->SetTitleOffset(1.6);
   ptframe_input->SetMinimum(0.1);
   ptframe_input->SetMaximum(1);
   ptframe_input->Draw();

   // Create legend description
   TLegend legend2(0.80,0.75,0.89,0.89);
   legend2.SetFillColor(kWhite);
   legend2.SetLineColor(kWhite);
   for(auto it_pt = samples_pt_dh.begin(); it_pt != samples_pt_dh.end(); ++it_pt){
     RooDataHist sample_pt_dh = *it_pt;
     legend2.AddEntry(sample_pt_dh.GetTitle(), sample_pt_dh.GetTitle(), "PE");
   }   
   legend2.Draw();

   c->cd(3);
   gPad->SetLeftMargin(0.15);
   etaframe_morph->GetYaxis()->SetTitleOffset(1.6);
   etaframe_morph->SetMinimum(0.1);
   etaframe_morph->SetMaximum(1);
   etaframe_morph->Draw();

   // Create legend description
   TLegend legend3(0.60,0.80,0.89,0.89);
   legend3.SetFillColor(kWhite);
   legend3.SetLineColor(kWhite);
   legend3.AddEntry("morph_eta","morphing function","L");
   legend3.AddEntry("validation_eta","validation sample","PE");
   legend3.Draw();

   c->cd(4);
   gPad->SetLeftMargin(0.15);
   ptframe_morph->GetYaxis()->SetTitleOffset(1.6);
   ptframe_morph->SetMinimum(0.1);
   ptframe_morph->SetMaximum(1);
   ptframe_morph->Draw();
   // Create legend description
   TLegend legend4(0.60,0.80,0.89,0.89);
   legend4.SetFillColor(kWhite);
   legend4.SetLineColor(kWhite);
   legend4.AddEntry("morph_pt","morphing function","L");
   legend4.AddEntry("validation_pt","validation sample","PE");
   legend4.Draw();
   legend4.Draw();
   c->SaveAs("rf1001.pdf");
   morphfunc_eta.Print();
}
