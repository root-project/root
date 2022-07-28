/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Morphing effective field theory distributions with RooLagrangianMorphFunc
/// A morphing function as a function of one coefficient is setup and can be used
/// to obtain the distribution for any value of the coefficient.
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \date January 2022
/// \author Rahul Balasubramanian

#include <RooAbsCollection.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooLagrangianMorphFunc.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include <TCanvas.h>
#include <TColor.h>
#include <TFile.h>
#include <TFolder.h>
#include <TH1.h>
#include <TLegend.h>
#include <TStyle.h>

using namespace RooFit;

void rf711_lagrangianmorph()
{
   // C r e a t e  v a r i a b l e s  f o r
   // m o r p h i n g  f u n c t i o n
   // ---------------------------------------------

   std::string observablename = "pTV";

   // Setup observable that is morphed
   RooRealVar obsvar(observablename.c_str(), "p_{T}^{V}", 10, 600);

   // Setup two couplings that enters the morphing function
   // kSM -> SM coupling set to constant (1)
   // cHq3 -> EFT parameter with NewPhysics attribute set to true
   RooRealVar kSM("kSM", "sm modifier", 1.0);
   RooRealVar cHq3("cHq3", "EFT modifier", 0.0, 1.0);
   cHq3.setAttribute("NewPhysics", true);

   // I n p u t s  n e e d e d  f o r  c o n f i g
   // ---------------------------------------------
   std::string infilename = std::string(gROOT->GetTutorialDir()) + "/roofit/input_histos_rf_lagrangianmorph.root";
   std::vector<std::string> samplelist = {"SM_NPsq0", "cHq3_NPsq1", "cHq3_NPsq2"};

   // S e t u p  C o n f i g
   // ---------------------------------------------
   RooLagrangianMorphFunc::Config config;
   config.fileName = infilename;
   config.observableName = observablename;
   config.folderNames = samplelist;
   config.couplings.add(cHq3);
   config.couplings.add(kSM);

   // C r e a t e  m o r p h i n g  f u n c t i o n
   // ---------------------------------------------
   RooLagrangianMorphFunc morphfunc("morphfunc", "morphed dist. of pTV", config);

   // G e t  m o r p h e d  d i s t r i b u t i o n  f o r
   // d i f f e r e n t  c H q 3
   // ---------------------------------------------
   morphfunc.setParameter("cHq3", 0.01);
   auto morph_hist_0p01 = morphfunc.createTH1("morph_cHq3=0.01");
   morphfunc.setParameter("cHq3", 0.25);
   auto morph_hist_0p25 = morphfunc.createTH1("morph_cHq3=0.25");
   morphfunc.setParameter("cHq3", 0.5);
   auto morph_hist_0p5 = morphfunc.createTH1("morph_cHq3=0.5");
   RooDataHist morph_datahist_0p01("morph_dh_cHq3=0.01", "", RooArgList(obsvar), morph_hist_0p01);
   RooDataHist morph_datahist_0p25("morph_dh_cHq3=0.25", "", RooArgList(obsvar), morph_hist_0p25);
   RooDataHist morph_datahist_0p5("morph_dh_cHq3=0.5", "", RooArgList(obsvar), morph_hist_0p5);

   // E x t r a c t  i n p u t  t e m p l a t e s
   // f o r  p l o t t i n g
   // ---------------------------------------------
   TFile *file = new TFile(infilename.c_str(), "READ");
   TFolder *folder = 0;
   file->GetObject(samplelist[0].c_str(), folder);
   TH1 *input_hist0 = dynamic_cast<TH1 *>(folder->FindObject(observablename.c_str()));
   input_hist0->SetDirectory(NULL);
   file->GetObject(samplelist[1].c_str(), folder);
   TH1 *input_hist1 = dynamic_cast<TH1 *>(folder->FindObject(observablename.c_str()));
   input_hist1->SetDirectory(NULL);
   file->GetObject(samplelist[2].c_str(), folder);
   TH1 *input_hist2 = dynamic_cast<TH1 *>(folder->FindObject(observablename.c_str()));
   input_hist2->SetDirectory(NULL);
   file->Close();

   RooDataHist input_dh0(samplelist[0].c_str(), "", RooArgList(obsvar), input_hist0);
   RooDataHist input_dh1(samplelist[1].c_str(), "", RooArgList(obsvar), input_hist1);
   RooDataHist input_dh2(samplelist[2].c_str(), "", RooArgList(obsvar), input_hist2);

   auto frame0 = obsvar.frame(Title("Input templates for p_{T}^{V}"));
   input_dh0.plotOn(frame0, Name(samplelist[0].c_str()), LineColor(kBlack), MarkerColor(kBlack), MarkerSize(1));
   input_dh1.plotOn(frame0, Name(samplelist[1].c_str()), LineColor(kRed), MarkerColor(kRed), MarkerSize(1));
   input_dh2.plotOn(frame0, Name(samplelist[2].c_str()), LineColor(kBlue), MarkerColor(kBlue), MarkerSize(1));

   // P l o t  m o r p h e d  d i s t r i b u t i o n  f o r
   // d i f f e r e n t  c H q 3
   // ---------------------------------------------
   auto frame1 = obsvar.frame(Title("Morphed templates for selected values"));
   morph_datahist_0p01.plotOn(frame1, Name("morph_dh_cHq3=0.01"), DrawOption("C"), LineColor(kGreen),
                              DataError(RooAbsData::None), XErrorSize(0));
   morph_datahist_0p25.plotOn(frame1, Name("morph_dh_cHq3=0.25"), DrawOption("C"), LineColor(kGreen + 1),
                              DataError(RooAbsData::None), XErrorSize(0));
   morph_datahist_0p5.plotOn(frame1, Name("morph_dh_cHq3=0.5"), DrawOption("C"), LineColor(kGreen + 2),
                             DataError(RooAbsData::None), XErrorSize(0));

   // C r e a t e  w r a p p e d  p d f  t o g e n e r a t e
   // 2D  d a t a s e t  o f  c H q 3  a s  a  f u n c t i o n  o f
   // o b s e r v a b l e
   // ---------------------------------------------

   RooWrapperPdf model("wrap_pdf", "wrap_pdf", morphfunc);
   RooDataSet *data = model.generate(RooArgSet(cHq3, obsvar), 1000000);
   TH1 *hh_data = data->createHistogram("pTV vs cHq3", obsvar, Binning(20), YVar(cHq3, Binning(50)));
   hh_data->SetTitle("Morphing prediction");

   // D r a w  p l o t s  o n  c a n v a s
   // ---------------------------------------------
   TCanvas *c1 = new TCanvas("fig3", "fig3", 1200, 400);
   c1->Divide(3, 1);

   c1->cd(1);
   gPad->SetLeftMargin(0.15);
   gPad->SetRightMargin(0.05);

   frame0->Draw();
   TLegend *leg1 = new TLegend(0.55, 0.65, 0.94, 0.87);
   leg1->SetTextSize(0.04);
   leg1->SetFillColor(kWhite);
   leg1->SetLineColor(kWhite);
   leg1->AddEntry(frame0->findObject("SM_NPsq0"), "SM", "LP");
   leg1->AddEntry((TObject *)0, "", "");
   leg1->AddEntry(frame0->findObject("cHq3_NPsq1"), "c_{Hq^(3)}=1.0 at #Lambda^{-2}", "LP");
   leg1->AddEntry((TObject *)0, "", "");
   leg1->AddEntry(frame0->findObject("cHq3_NPsq2"), "c_{Hq^(3)}=1.0 at #Lambda^{-4}", "LP");
   leg1->Draw();

   c1->cd(2);
   gPad->SetLeftMargin(0.15);
   gPad->SetRightMargin(0.05);

   frame1->Draw();

   TLegend *leg2 = new TLegend(0.60, 0.65, 0.94, 0.87);
   leg2->SetTextSize(0.04);
   leg2->SetFillColor(kWhite);
   leg2->SetLineColor(kWhite);
   leg2->AddEntry(frame1->findObject("morph_dh_cHq3=0.01"), "c_{Hq^{(3)}}=0.01", "L");
   leg2->AddEntry((TObject *)0, "", "");
   leg2->AddEntry(frame1->findObject("morph_dh_cHq3=0.25"), "c_{Hq^{(3)}}=0.25", "L");
   leg2->AddEntry((TObject *)0, "", "");
   leg2->AddEntry(frame1->findObject("morph_dh_cHq3=0.5"), "c_{Hq^{(3)}}=0.5", "L");
   leg2->AddEntry((TObject *)0, "", "");
   leg2->Draw();

   c1->cd(3);
   gPad->SetLeftMargin(0.12);
   gPad->SetRightMargin(0.18);
   gStyle->SetNumberContours(255);
   gStyle->SetPalette(kGreyScale);
   gStyle->SetOptStat(0);
   TColor::InvertPalette();
   gPad->SetLogz();
   hh_data->GetYaxis()->SetTitle("c_{Hq^{(3)}}");
   hh_data->GetYaxis()->SetRangeUser(0, 0.5);
   hh_data->GetZaxis()->SetTitleOffset(1.8);
   hh_data->Draw("COLZ");
   c1->SaveAs("rf711_lagrangianmorph.png");
}
