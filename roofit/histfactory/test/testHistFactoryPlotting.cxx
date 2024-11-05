// Tests for the ParamHistFunc
// Authors: Jonas Rembser, CERN  08/2023

#include <RooStats/HistFactory/Measurement.h>
#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>
#include <RooStats/HistFactory/Sample.h>
#include <RooFit/ModelConfig.h>

#include <RooCategory.h>
#include <RooWorkspace.h>
#include <RooSimultaneous.h>
#include <RooRealVar.h>
#include <RooHelpers.h>
#include <RooPlot.h>

#include <TROOT.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TH1D.h>

#include <gtest/gtest.h>

void createToyHistos1D()
{
   // Save histograms in this root file
   std::unique_ptr<TFile> histoFile{TFile::Open("histoFile.root", "RECREATE")};

   // Set number of events in signal and background (N_Sig = N_BG = N_DATA/2)
   int nEvents = 10000;

   // Functions used to fill histograms
   TF1 func1{"func1", "1 + 0.01 * x", 3000, 6000};                                         // background
   TF1 func2{"func2", "0.2 * exp(-( (x-4000.)*(x-4000.)/(2.0*200.*200.) ) )", 3000, 6000}; // signal

   // Histograms
   TH1D *histoBG = new TH1D("histoBG", "background histo", 15, 3000, 6000);
   TH1D *histoSig = new TH1D("histoSig", "signal histo", 15, 3000, 6000);
   TH1D *histoData = new TH1D("histoData", "data histo", 15, 3000, 6000);

   // define observable x
   double x_bg;  // background
   double x_sig; // signal

   // Fill background and signal histograms
   for (int i = 0; i < nEvents; ++i) {
      x_bg = func1.GetRandom();
      x_sig = func2.GetRandom();
      histoBG->Fill(x_bg);
      histoSig->Fill(x_sig);
   }

   // Fill data histogram
   for (int i = 0; i < nEvents; ++i) {
      x_bg = func1.GetRandom();
      histoData->Fill(x_bg);
      x_sig = func2.GetRandom();
      histoData->Fill(x_sig);
   }

   histoFile->Write();
}

/// Test that plotting HistFactory components works correctly. Covers the
/// problem reported in this forum post:
/// https://root-forum.cern.ch/t/problems-plotting-individual-components-with-roofit-histfactory
TEST(HistFactoryPlotting, ComponentSelection)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   using namespace RooFit;

   // Make histograms
   createToyHistos1D();

   // --------------------------------------
   // Load histograms
   // --------------------------------------

   // file with histograms
   std::string histoFileName = "histoFile.root";

   // load histograms
   std::unique_ptr<TFile> histoFile{TFile::Open(histoFileName.c_str(), "READ")};
   std::vector<TH1 *> templateHistos;
   templateHistos.emplace_back(histoFile->Get<TH1>("histoSig"));
   templateHistos.emplace_back(histoFile->Get<TH1>("histoBG"));

   // --------------------------------------
   // Setup measurement & channel
   // --------------------------------------

   // -- Define the measurement
   RooStats::HistFactory::Measurement meas("B2D0MuNu", "B2D0MuNu fit");

   meas.SetPOI("num_histoSig"); // set to Bogus parma. of interest
   meas.SetLumi(1.0);
   meas.SetLumiRelErr(0.1);

   // -- Define the channel (in our case, there is only one)
   RooStats::HistFactory::Channel B2D0MuNu("BMCorr");
   B2D0MuNu.SetData("histoData", "histoFile.root");

   // -- Add the histograms to the channel
   for (auto const &histo : templateHistos) {

      RooStats::HistFactory::Sample sample(histo->GetName(), histo->GetName(), histoFileName);

      sample.SetNormalizeByTheory(false);
      // fix normalisation
      sample.AddNormFactor("norm_" + std::string{histo->GetName()}, 1 / histo->Integral(), 1 / histo->Integral(),
                           1 / histo->Integral());
      // free yield
      sample.AddNormFactor("num_" + std::string{histo->GetName()}, histo->Integral(), 1.0, histo->Integral() * 2);

      // add histograms to channel
      B2D0MuNu.AddSample(sample);
   }

   // add channel to measurements
   meas.AddChannel(B2D0MuNu);
   meas.CollectHistograms();

   // --------------------------------------
   // Setting up workspace
   // --------------------------------------

   // -- Define workspace
   std::unique_ptr<RooWorkspace> ws{RooStats::HistFactory::MakeModelAndMeasurementFast(meas)};

   // Get model manually
   auto *modelConf = static_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   auto *model = static_cast<RooSimultaneous *>(modelConf->GetPdf());

   // fix the Lumi to not have it also fitted
   RooArgSet const *nuis = modelConf->GetNuisanceParameters();
   (static_cast<RooRealVar *>(nuis->find("Lumi")))->setConstant(true);
   (static_cast<RooRealVar *>(nuis->find("norm_histoSig")))->setConstant(true); // fix norm
   (static_cast<RooRealVar *>(nuis->find("norm_histoBG")))->setConstant(true);  // fix norm

   auto *obs = static_cast<RooArgSet const *>(modelConf->GetObservables());
   auto *idx = static_cast<RooCategory *>(obs->find("channelCat"));
   RooAbsData *data = ws->data("obsData");
   auto *obs_x = static_cast<RooRealVar *>(obs->find("obs_x_BMCorr"));

   // plot data, total pdf and individual components
   std::unique_ptr<RooPlot> frame{obs_x->frame()};
   data->plotOn(frame.get(), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"), Name("pdf_Data"));
   model->plotOn(frame.get(), ProjWData(*idx, *data), Name("pdf_model"), LineColor(kGreen));
   model->plotOn(frame.get(), ProjWData(*idx, *data), Name("pdf_sig"), Components("*histoSig*"), LineColor(kBlue));
   model->plotOn(frame.get(), ProjWData(*idx, *data), Name("pdf_bg"), Components("*histoBG*"), LineColor(kRed));

   auto *curveFull = static_cast<RooCurve *>(frame->findObject("pdf_model"));
   auto *curgeSig = static_cast<RooCurve *>(frame->findObject("pdf_sig"));
   auto *curgeBg = static_cast<RooCurve *>(frame->findObject("pdf_bg"));

   for (int i = 0; i < curveFull->GetN(); ++i) {
      double x1;
      double x2;
      double x3;
      double y1;
      double y2;
      double y3;
      curveFull->GetPoint(i, x1, y1);
      curgeSig->GetPoint(i, x2, y2);
      curgeBg->GetPoint(i, x3, y3);
      // The points for the components should add up to the points of the full model.
      EXPECT_DOUBLE_EQ(y2 + y3, y1);
   }

   constexpr bool makePlot = false;

   if (makePlot) {
      gROOT->SetBatch(true);
      TCanvas c1{};
      frame->Draw();
      c1.SaveAs("hf_component_plot.png");
   }
}
