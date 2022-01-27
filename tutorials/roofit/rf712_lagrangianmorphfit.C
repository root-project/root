/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Performing a simple fit with RooLagrangianMorphFunc.
/// a morphing function is setup as a function of three variables and
/// a fit is performed on a pseudo-dataset.
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \date January 2022
/// \author Rahul Balasubramanian

#include <RooDataHist.h>
#include <RooFitResult.h>
#include <RooLagrangianMorphFunc.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include <TAxis.h>
#include <TCanvas.h>
#include <TH2.h>
#include <TStyle.h>

using namespace RooFit;

void rf712_lagrangianmorphfit()
{
   // C r e a t e  v a r i a b l e s  f o r
   // m o r p h i n g  f u n c t i o n
   // ---------------------------------------------

   std::string observablename = "pTV";
   RooRealVar obsvar(observablename.c_str(), "observable of pTV", 10, 600);
   RooRealVar kSM("kSM", "sm modifier", 1.0);
   RooRealVar cHq3("cHq3", "EFT modifier", -10.0, 10.0);
   cHq3.setAttribute("NewPhysics", true);
   RooRealVar cHl3("cHl3", "EFT modifier", -10.0, 10.0);
   cHl3.setAttribute("NewPhysics", true);
   RooRealVar cHDD("cHDD", "EFT modifier", -10.0, 10.0);
   cHDD.setAttribute("NewPhysics", true);

   // I n p u t s  n e e d e d  f o r  c o n f i g
   // ---------------------------------------------
   std::string infilename = std::string(gROOT->GetTutorialDir()) + "/roofit/input_histos_rf_lagrangianmorph.root";
   std::vector<std::string> samplelist = {"SM_NPsq0",        "cHq3_NPsq1",     "cHq3_NPsq2", "cHl3_NPsq1",
                                          "cHl3_NPsq2",      "cHDD_NPsq1",     "cHDD_NPsq2", "cHl3_cHDD_NPsq2",
                                          "cHq3_cHDD_NPsq2", "cHl3_cHq3_NPsq2"};

   // S e t u p  C o n f i g
   // ---------------------------------------------
   RooLagrangianMorphFunc::Config config;
   config.fileName = infilename;
   config.observableName = observablename;
   config.folderNames = samplelist;
   config.couplings.add(cHq3);
   config.couplings.add(cHl3);
   config.couplings.add(cHDD);
   config.couplings.add(kSM);

   // C r e a t e  m o r p h i n g  f u n c t i o n
   // ---------------------------------------------
   RooLagrangianMorphFunc morphfunc("morphfunc", "morphed dist. of pTV", config);

   // C r e a t e  p s e u d o  d a t a  h i s t o g r a m
   // f o r  f i t
   // ---------------------------------------------
   morphfunc.setParameter("cHq3", 0.01);
   morphfunc.setParameter("cHl3", 1.0);
   morphfunc.setParameter("cHDD", 0.2);

   auto pseudo_hist = morphfunc.createTH1("pseudo_hist");
   auto pseudo_dh = new RooDataHist("pseudo_dh", "pseudo_dh", RooArgList(obsvar), pseudo_hist);

   // reset parameters to zeros before fit
   morphfunc.setParameter("cHq3", 0.0);
   morphfunc.setParameter("cHl3", 0.0);
   morphfunc.setParameter("cHDD", 0.0);

   // error set used as initial step size
   cHq3.setError(0.1);
   cHl3.setError(0.1);
   cHDD.setError(0.1);

   // W r a p  p d f  o n  m o r p h f u n c  a n d
   // f i t  t o  d a t a  h i s t o g r a m
   // ---------------------------------------------
   // wrapper pdf to normalise morphing function to a morphing pdf
   RooWrapperPdf model("wrap_pdf", "wrap_pdf", morphfunc);
   auto fitres = model.fitTo(*pseudo_dh, SumW2Error(true), Optimize(false), Save());
   auto hcorr = fitres->correlationHist();

   // E x t r a c t  p o s t f i t  d i s t r i b u t i o n
   // a n d  p l o t  w i t h  i n i t i a l
   // h i s t o g r a m
   // ---------------------------------------------
   auto postfit_hist = morphfunc.createTH1("morphing_postfit_hist");
   RooDataHist postfit_dh("morphing_postfit_dh", "morphing_postfit_dh", RooArgList(obsvar), postfit_hist);

   auto frame0 = obsvar.frame(Title("Fitted histogram of p_{T}^{V}"));
   postfit_dh.plotOn(frame0, Name("postfit_dist"), DrawOption("C"), LineColor(kBlue), DataError(RooAbsData::None),
                     XErrorSize(0));
   pseudo_dh->plotOn(frame0, Name("input"));

   // D r a w  p l o t s  o n  c a n v a s
   // ---------------------------------------------
   TCanvas *c1 = new TCanvas("fig3", "fig3", 800, 400);
   c1->Divide(2, 1);

   c1->cd(1);
   gPad->SetLeftMargin(0.15);
   gPad->SetRightMargin(0.05);

   model.paramOn(frame0, Layout(0.50, 0.75, 0.9), Parameters(config.couplings));
   frame0->GetXaxis()->SetTitle("p_{T}^{V}");
   frame0->Draw();

   c1->cd(2);
   gPad->SetLeftMargin(0.15);
   gPad->SetRightMargin(0.15);
   gStyle->SetPaintTextFormat("4.1f");
   gStyle->SetOptStat(0);
   hcorr->SetMarkerSize(3.);
   hcorr->SetTitle("correlation matrix");
   hcorr->GetYaxis()->SetTitleOffset(1.4);
   hcorr->GetYaxis()->SetLabelSize(0.1);
   hcorr->GetXaxis()->SetLabelSize(0.1);
   hcorr->GetYaxis()->SetBinLabel(1, "c_{HDD}");
   hcorr->GetYaxis()->SetBinLabel(2, "c_{Hl^{(3)}}");
   hcorr->GetYaxis()->SetBinLabel(3, "c_{Hq^{(3)}}");
   hcorr->GetXaxis()->SetBinLabel(3, "c_{HDD}");
   hcorr->GetXaxis()->SetBinLabel(2, "c_{Hl^{(3)}}");
   hcorr->GetXaxis()->SetBinLabel(1, "c_{Hq^{(3)}}");
   hcorr->Draw("colz text");
   c1->SaveAs("rf712_lagrangianmorphfit.png");
}
