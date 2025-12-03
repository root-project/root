// Tests for the HistFactory
// Authors: Stephan Hageboeck, CERN  01/2019
//          Jonas Rembser, CERN  06/2023

#include <RooStats/HistFactory/Measurement.h>
#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>
#include <RooStats/HistFactory/Sample.h>
#include <RooFit/ModelConfig.h>

#include <RooFitHS3/JSONIO.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooFit/Detail/NormalizationHelpers.h>
#include <RooDataHist.h>
#include <RooWorkspace.h>
#include <RooArgSet.h>
#include <RooSimultaneous.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooHelpers.h>
#include <RooFitResult.h>
#include <RooPlot.h>
#include <RooFit/Evaluator.h>

#include <TROOT.h>
#include <TFile.h>
#include <TCanvas.h>
#include <gtest/gtest.h>

#include "../../roofitcore/test/gtest_wrapper.h"

#include <set>

namespace {

// If the JSON files should be written out for debugging purpose.
const bool writeJsonFiles = false;

std::vector<double> getValues(RooAbsReal const &real, RooRealVar &obs, bool normalize, bool useBatchMode)
{
   RooArgSet normSet{obs};

   std::vector<double> out;
   // We want to evaluate the function at the bin centers
   std::vector<double> binCenters(obs.numBins());
   for (int iBin = 0; iBin < obs.numBins(); ++iBin) {
      obs.setBin(iBin);
      binCenters[iBin] = obs.getVal();
      out.push_back(normalize ? real.getVal(normSet) : real.getVal());
   }

   if (useBatchMode == false) {
      return out;
   }

   std::unique_ptr<RooAbsReal> clone;
   if (normalize) {
      clone = RooFit::Detail::compileForNormSet<RooAbsReal>(real, obs);
   } else {
      clone.reset(static_cast<RooAbsReal *>(real.cloneTree()));
   }

   RooFit::Evaluator evaluator(*clone);
   evaluator.setInput(obs.GetName(), binCenters, false);
   std::span<const double> results = evaluator.run();
   out.assign(results.begin(), results.end());
   return out;
}

} // namespace

TEST(Sample, CopyAssignment)
{
   RooStats::HistFactory::Sample s("s");
   {
      RooStats::HistFactory::Sample s1("s1");
      auto hist1 = new TH1D("hist1", "hist1", 10, 0, 10);
      s1.SetHisto(hist1);
      s = s1;
      // Now go out of scope. Should delete hist1, that's owned by s1.
   }

   auto hist = s.GetHisto();
   ASSERT_EQ(hist->GetNbinsX(), 10);
}

TEST(HistFactory, Read_ROOT6_16_Model)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

   std::string filename = "./ref_6.16_example_UsingC_channel1_meas_model.root";
   std::unique_ptr<TFile> file(TFile::Open(filename.c_str()));
   if (!file || !file->IsOpen()) {
      filename = TROOT::GetRootSys() + "/roofit/histfactory/test/" + filename;
      file.reset(TFile::Open(filename.c_str()));
   }

   ASSERT_TRUE(file && file->IsOpen());
   RooWorkspace *ws;
   file->GetObject("channel1", ws);
   ASSERT_NE(ws, nullptr);

   auto mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   ASSERT_NE(mc, nullptr);

   RooAbsPdf *pdf = mc->GetPdf();
   ASSERT_NE(pdf, nullptr);

   const RooArgSet *obs = mc->GetObservables();
   ASSERT_NE(obs, nullptr);

   EXPECT_NEAR(pdf->getVal(), 0.17488817, 1.E-8);
   EXPECT_NEAR(pdf->getVal(*obs), 0.95652174, 1.E-8);
}

TEST(HistFactory, Read_ROOT6_16_Combined_Model)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

   std::string filename = "./ref_6.16_example_UsingC_combined_meas_model.root";
   std::unique_ptr<TFile> file(TFile::Open(filename.c_str()));
   if (!file || !file->IsOpen()) {
      filename = TROOT::GetRootSys() + "/roofit/histfactory/test/" + filename;
      file.reset(TFile::Open(filename.c_str()));
   }

   ASSERT_TRUE(file && file->IsOpen());
   RooWorkspace *ws;
   file->GetObject("combined", ws);
   ASSERT_NE(ws, nullptr);

   auto mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   ASSERT_NE(mc, nullptr);

   RooAbsPdf *pdf = mc->GetPdf();
   ASSERT_NE(pdf, nullptr);

   const RooArgSet *obs = mc->GetObservables();
   ASSERT_NE(obs, nullptr);

   EXPECT_NEAR(pdf->getVal(), 0.17488817, 1.E-8);
   EXPECT_NEAR(pdf->getVal(*obs), 0.95652174, 1.E-8);
}

/// What kind of model is set up. Use this to instantiate
/// a test suite.
enum class MakeModelMode { OverallSyst, HistoSyst, StatSyst, ShapeSyst };

using HFTestParam = std::tuple<MakeModelMode, bool, RooFit::EvalBackend>;

std::string getName(HFTestParam const &param, bool ignoreBackend = false)
{
   const MakeModelMode mode = std::get<0>(param);
   const bool customBins = std::get<1>(param);
   auto const &evalBackend = std::get<2>(param);

   std::stringstream ss;

   ss << (customBins ? "CustomBins" : "EquidistantBins");

   if (mode == MakeModelMode::OverallSyst)
      ss << "_OverallSyst";
   if (mode == MakeModelMode::HistoSyst)
      ss << "_HistoSyst";
   if (mode == MakeModelMode::StatSyst)
      ss << "_StatSyst";
   if (mode == MakeModelMode::ShapeSyst)
      ss << "_ShapeSyst";

   if (!ignoreBackend) {
      ss << "_Backend_" << evalBackend.name();
   }

   return ss.str();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Fixture class to set up a toy hist factory model.
/// In the SetUp() phase
/// - A file with histograms is created. Depending on the value of MakeModelMode,
/// equidistant or custom bins are used, and shape systematics are added.
/// - A Measurement with the histograms in the file is created.
/// - The corresponding workspace is created.
class HFFixture : public testing::TestWithParam<HFTestParam> {
public:
   std::string _inputFile{"TestMakeModel.root"};
   static constexpr bool _verbose = false;
   double _customBins[3] = {0., 1.8, 2.};
   const double _tgtMu = 2.;
   const double _tgtNom[2] = {110., 120.};
   const double _tgtSysUp[2] = {112., 140.};
   const double _tgtSysDo[2] = {108., 100.};
   const double _tgtShapeSystBkg1[2] = {0.15, 0.0};
   const double _tgtShapeSystBkg2[2] = {0.0, 0.10};
   std::unique_ptr<RooWorkspace> ws;
   std::set<std::string> _systNames; // Systematics defined during set up
   std::unique_ptr<RooStats::HistFactory::Measurement> _measurement;
   std::string _name;

   TH1D *createHisto(const char *name, const char *title, bool customBins)
   {
      if (customBins)
         return new TH1D{name, title, 2, _customBins};
      return new TH1D{name, title, 2, 1, 2};
   }

   void SetUp() override
   {
      using namespace RooStats::HistFactory;

      _name = getName(GetParam());

      const MakeModelMode makeModelMode = std::get<0>(GetParam());
      const bool customBins = std::get<1>(GetParam());

      {
         TFile example(_inputFile.c_str(), "RECREATE");

         TH1D *data = createHisto("data", "data", customBins);
         TH1D *signal = createHisto("signal", "signal histogram (pb)", customBins);
         TH1D *bkg1 = createHisto("background1", "background 1 histogram (pb)", customBins);
         TH1D *bkg2 = createHisto("background2", "background 2 histogram (pb)", customBins);
         TH1D *statUnc = createHisto("background1_statUncert", "statUncert", customBins);
         TH1D *systUncUp = createHisto("histoUnc_sigUp", "signal shape uncert.", customBins);
         TH1D *systUncDo = createHisto("histoUnc_sigDo", "signal shape uncert.", customBins);
         TH1D *shapeSystBkg1 = createHisto("background1_shapeSyst", "background 1 shapeSyst", customBins);
         TH1D *shapeSystBkg2 = createHisto("background2_shapeSyst", "background 2 shapeSyst", customBins);

         bkg1->SetBinContent(1, 100);
         bkg2->SetBinContent(2, 100);

         for (unsigned int bin = 0; bin < 2; ++bin) {
            signal->SetBinContent(bin + 1, _tgtNom[bin] - 100.);
            systUncUp->SetBinContent(bin + 1, _tgtSysUp[bin] - 100.);
            systUncDo->SetBinContent(bin + 1, _tgtSysDo[bin] - 100.);
            shapeSystBkg1->SetBinContent(bin + 1, _tgtShapeSystBkg1[bin]);
            shapeSystBkg2->SetBinContent(bin + 1, _tgtShapeSystBkg2[bin]);

            if (makeModelMode == MakeModelMode::OverallSyst) {
               data->SetBinContent(bin + 1, _tgtMu * signal->GetBinContent(bin + 1) + 100.);
            } else if (makeModelMode == MakeModelMode::HistoSyst) {
               // Set data such that alpha = -1., fit should pull parameter.
               data->SetBinContent(bin + 1, _tgtMu * systUncDo->GetBinContent(bin + 1) + 100.);
            } else if (makeModelMode == MakeModelMode::StatSyst) {
               // Tighten the stat. errors of the model, and kick bin 0, so the gammas have to adapt
               signal->SetBinError(bin + 1, 0.1 * std::sqrt(signal->GetBinContent(bin + 1)));
               bkg1->SetBinError(bin + 1, 0.1 * std::sqrt(bkg1->GetBinContent(bin + 1)));
               bkg2->SetBinError(bin + 1, 0.1 * std::sqrt(bkg2->GetBinContent(bin + 1)));

               data->SetBinContent(bin + 1, _tgtMu * signal->GetBinContent(bin + 1) + 100. + (bin == 0 ? 50. : 0.));
            } else if (makeModelMode == MakeModelMode::ShapeSyst) {
               // Distort data such that the shape systematics will pull gamma
               // down in one bin and up in the other.
               data->SetBinContent(bin + 1, _tgtMu * signal->GetBinContent(bin + 1) + (bin == 0 ? 85. : 110));
            }

            // A small statistical uncertainty
            statUnc->SetBinContent(bin + 1, .05); // 5% uncertainty
         }

         for (auto hist : {data, signal, bkg1, bkg2, statUnc, systUncUp, systUncDo, shapeSystBkg1, shapeSystBkg2}) {
            example.WriteTObject(hist);
         }
      }

      // Create the measurement
      _measurement = std::make_unique<RooStats::HistFactory::Measurement>("meas", "meas");
      RooStats::HistFactory::Measurement &meas = *_measurement;

      meas.SetOutputFilePrefix("example_variableBins");
      meas.SetPOI("SigXsecOverSM");
      meas.AddConstantParam("Lumi");
      if (makeModelMode == MakeModelMode::HistoSyst || makeModelMode == MakeModelMode::ShapeSyst) {
         meas.AddConstantParam("gamma_stat_channel1_bin_0");
         meas.AddConstantParam("gamma_stat_channel1_bin_1");
      }

      meas.SetLumi(1.0);
      meas.SetLumiRelErr(0.10);

      // Create a channel
      RooStats::HistFactory::Channel chan("channel1");
      chan.SetData("data", _inputFile);
      chan.SetStatErrorConfig(0.005, "Poisson");
      _systNames.insert("gamma_stat_channel1_bin_0");
      _systNames.insert("gamma_stat_channel1_bin_1");

      // Now, create some samples

      // Create the signal sample
      RooStats::HistFactory::Sample signal("signal", "signal", _inputFile);

      signal.AddNormFactor("SigXsecOverSM", 1, 0, 3);
      if (makeModelMode == MakeModelMode::HistoSyst) {
         signal.AddHistoSys("SignalShape", "histoUnc_sigDo", _inputFile, "", "histoUnc_sigUp", _inputFile, "");
         _systNames.insert("alpha_SignalShape");
      }
      chan.AddSample(signal);

      // Background 1
      RooStats::HistFactory::Sample background1("background1", "background1", _inputFile);
      background1.ActivateStatError("background1_statUncert", _inputFile);
      if (makeModelMode == MakeModelMode::OverallSyst) {
         background1.AddOverallSys("syst2", 0.95, 1.05);
         background1.AddOverallSys("syst3", 0.99, 1.01);
         _systNames.insert("alpha_syst2");
         _systNames.insert("alpha_syst3");
      }
      if (makeModelMode == MakeModelMode::ShapeSyst) {
         background1.AddShapeSys("background1Shape", Constraint::Gaussian, "background1_shapeSyst", _inputFile);
         meas.AddConstantParam("gamma_background1Shape_bin_1");
         _systNames.insert("gamma_background1Shape_bin_0");
         _systNames.insert("gamma_background1Shape_bin_1");
      }
      chan.AddSample(background1);

      // Background 2
      RooStats::HistFactory::Sample background2("background2", "background2", _inputFile);
      background2.ActivateStatError();
      if (makeModelMode == MakeModelMode::OverallSyst) {
         background2.AddOverallSys("syst3", 0.99, 1.01);
         background2.AddOverallSys("syst4", 0.95, 1.05);
         _systNames.insert("alpha_syst3");
         _systNames.insert("alpha_syst4");
      }
      if (makeModelMode == MakeModelMode::ShapeSyst) {
         background2.AddShapeSys("background2Shape", Constraint::Poisson, "background2_shapeSyst", _inputFile);
         meas.AddConstantParam("gamma_background2Shape_bin_0");
         _systNames.insert("gamma_background2Shape_bin_0");
         _systNames.insert("gamma_background2Shape_bin_1");
      }
      chan.AddSample(background2);

      // Done with this channel
      // Add it to the measurement:
      meas.AddChannel(chan);

      if (!_verbose) {
         RooMsgService::instance().getStream(1).minLevel = RooFit::PROGRESS;
         RooMsgService::instance().getStream(2).minLevel = RooFit::WARNING;
      }
      RooHelpers::HijackMessageStream hijackW(RooFit::WARNING, RooFit::HistFactory);

      // Collect the histograms from their files,
      meas.CollectHistograms();

      // Now, create the measurement
      ws = std::unique_ptr<RooWorkspace>(MakeModelAndMeasurementFast(meas));

      EXPECT_TRUE(hijackW.str().empty()) << "Warnings logged for HistFactory:\n" << hijackW.str();
   }

   void TearDown() override {}
};

class HFFixtureEval : public HFFixture {};

class HFFixtureFit : public HFFixture {};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Test that the model consists of what is expected
TEST_P(HFFixture, ModelProperties)
{
   const MakeModelMode makeModelMode = std::get<0>(GetParam());
   const bool customBins = std::get<1>(GetParam());

   auto simPdf = dynamic_cast<RooSimultaneous *>(ws->pdf("simPdf"));
   ASSERT_NE(simPdf, nullptr);

   auto channelPdf = dynamic_cast<RooRealSumPdf *>(ws->pdf("channel1_model"));
   ASSERT_NE(channelPdf, nullptr);

   auto obs = dynamic_cast<RooRealVar *>(ws->var("obs_x_channel1"));

   // Nice to inspect the model if needed:
   // channelPdf->graphVizTree("graphVizTree.dot");

   // Check bin widths
   ASSERT_NE(obs, nullptr);
   if (!customBins) {
      EXPECT_DOUBLE_EQ(obs->getBinWidth(0), 0.5);
      EXPECT_DOUBLE_EQ(obs->getBinWidth(1), 0.5);
      EXPECT_EQ(obs->numBins(), 2);
   } else {
      EXPECT_DOUBLE_EQ(obs->getBinWidth(0), _customBins[1] - _customBins[0]);
      EXPECT_DOUBLE_EQ(obs->getBinWidth(1), _customBins[2] - _customBins[1]);
      EXPECT_EQ(obs->numBins(), 2);
   }

   auto mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   ASSERT_NE(mc, nullptr);

   // Check that parameters are in the model
   for (const auto &systName : _systNames) {
      auto &var = *ws->var(systName);

      EXPECT_TRUE(channelPdf->dependsOnValue(var)) << "Expect channel pdf to depend on " << systName;
      if (!var.isConstant()) {
         EXPECT_NE(mc->GetNuisanceParameters()->find(systName.c_str()), nullptr)
            << systName << " should be in list of nuisance parameters.";
      }
   }

   // Check that sub models depend on their systematic uncertainties.
   for (auto &subModelName : std::initializer_list<std::string>{"signal_channel1_shapes", "background1_channel1_shapes",
                                                                "background2_channel1_shapes"}) {
      auto subModel = ws->function(subModelName);
      ASSERT_NE(subModel, nullptr) << "Unable to retrieve sub model with name " << subModelName;
      if (subModelName.find("signal") != std::string::npos) {
         EXPECT_FALSE(subModel->dependsOn(*ws->var("gamma_stat_channel1_bin_0")));
         EXPECT_FALSE(subModel->dependsOn(*ws->var("gamma_stat_channel1_bin_1")));
      } else {
         EXPECT_TRUE(subModel->dependsOn(*ws->var("gamma_stat_channel1_bin_0")));
         EXPECT_TRUE(subModel->dependsOn(*ws->var("gamma_stat_channel1_bin_1")));
      }
   }

   EXPECT_TRUE(ws->function("signal_channel1_scaleFactors")->dependsOn(*ws->var("SigXsecOverSM")));
   EXPECT_FALSE(ws->function("background1_channel1_scaleFactors")->dependsOn(*ws->var("SigXsecOverSM")));
   EXPECT_FALSE(ws->function("background2_channel1_scaleFactors")->dependsOn(*ws->var("SigXsecOverSM")));

   if (makeModelMode == MakeModelMode::OverallSyst) {
      EXPECT_FALSE(ws->function("signal_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst2")));
      EXPECT_FALSE(ws->function("signal_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst3")));
      EXPECT_FALSE(ws->function("signal_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst4")));

      EXPECT_TRUE(ws->function("background1_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst2")));
      EXPECT_TRUE(ws->function("background1_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst3")));
      EXPECT_FALSE(ws->function("background1_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst4")));

      EXPECT_FALSE(ws->function("background2_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst2")));
      EXPECT_TRUE(ws->function("background2_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst3")));
      EXPECT_TRUE(ws->function("background2_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst4")));
   }

   EXPECT_EQ(*mc->GetParametersOfInterest()->begin(), ws->var("SigXsecOverSM"));
}

/// Test that the values returned are as expected.
TEST_P(HFFixtureEval, Evaluation)
{
   const double defaultEps = 1e-9;
   const double systEps = 1e-6;

   const MakeModelMode makeModelMode = std::get<0>(GetParam());
   const bool useBatchMode = std::get<2>(GetParam()) != RooFit::EvalBackend::Legacy();

   RooHelpers::HijackMessageStream evalMessages(RooFit::INFO, RooFit::FastEvaluations);

   auto simPdf = dynamic_cast<RooSimultaneous *>(ws->pdf("simPdf"));
   ASSERT_NE(simPdf, nullptr);

   auto channelPdf = dynamic_cast<RooRealSumPdf *>(ws->pdf("channel1_model"));
   ASSERT_NE(channelPdf, nullptr);

   auto obs = dynamic_cast<RooRealVar *>(ws->var("obs_x_channel1"));

   auto mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   ASSERT_NE(mc, nullptr);

   // Test evaluating the model:
   std::vector<double> normResults = getValues(*channelPdf, *obs, true, useBatchMode);
   std::vector<double> unnormResults = getValues(*channelPdf, *obs, false, useBatchMode);

   for (int i = 0; i < obs->numBins(); ++i) {
      EXPECT_NEAR(unnormResults[i], _tgtNom[i] / obs->getBinWidth(i), defaultEps);
      EXPECT_NEAR(normResults[i], _tgtNom[i] / obs->getBinWidth(i) / (_tgtNom[0] + _tgtNom[1]), defaultEps);
   }
   EXPECT_NEAR(normResults[0] * obs->getBinWidth(0) + normResults[1] * obs->getBinWidth(1), 1, defaultEps)
      << "Integral over PDF range should be 1.";

   // Test that shape uncertainties have an effect:
   if (makeModelMode == MakeModelMode::HistoSyst) {
      auto var = ws->var("alpha_SignalShape");
      ASSERT_NE(var, nullptr);

      // Test syst up:
      var->setVal(1.);
      std::vector<double> normResultsSyst = getValues(*channelPdf, *obs, true, useBatchMode);
      std::vector<double> unnormResultsSyst = getValues(*channelPdf, *obs, false, useBatchMode);
      for (int i = 0; i < obs->numBins(); ++i) {
         EXPECT_NEAR(unnormResultsSyst[i], _tgtSysUp[i] / obs->getBinWidth(i), systEps);
         EXPECT_NEAR(normResultsSyst[i], _tgtSysUp[i] / obs->getBinWidth(i) / (_tgtSysUp[0] + _tgtSysUp[1]), systEps);
      }

      // Test syst down:
      var->setVal(-1.);
      normResultsSyst = getValues(*channelPdf, *obs, true, useBatchMode);
      unnormResultsSyst = getValues(*channelPdf, *obs, false, useBatchMode);
      for (int i = 0; i < obs->numBins(); ++i) {
         EXPECT_NEAR(unnormResultsSyst[i], _tgtSysDo[i] / obs->getBinWidth(i), systEps);
         EXPECT_NEAR(normResultsSyst[i], _tgtSysDo[i] / obs->getBinWidth(i) / (_tgtSysDo[0] + _tgtSysDo[1]), systEps);
      }
   }

   EXPECT_TRUE(evalMessages.str().empty()) << "RooFit issued " << evalMessages.str().substr(0, 1000) << " [...]";
}

void setInitialFitParameters(RooWorkspace &ws, MakeModelMode makeModelMode)
{
   if (makeModelMode == MakeModelMode::OverallSyst) {
      // The final parameters of alpha_syst2 and alpha_syst4 are very close to the
      // pre-fit value zero. For the fit to converge reliably, the pre-fit values
      // are set away from the minimum.
      ws.var("alpha_syst2")->setVal(1.0);
      ws.var("alpha_syst4")->setVal(-1.0);
   }
   if (makeModelMode == MakeModelMode::ShapeSyst) {
      ws.var("gamma_background1Shape_bin_0")->setVal(0.7);
      ws.var("gamma_background2Shape_bin_1")->setVal(1.3);
   }
}

TEST_P(HFFixture, HS3ClosureLoop)
{
   const MakeModelMode makeModelMode = std::get<0>(GetParam());

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   EXPECT_TRUE(mc != nullptr);

   RooAbsPdf *pdf = mc->GetPdf();
   EXPECT_TRUE(pdf != nullptr);

   std::string const &js = RooJSONFactoryWSTool{*ws}.exportJSONtoString();
   if (writeJsonFiles) {
      RooJSONFactoryWSTool{*ws}.exportJSON(_name + "_1.json");
   }

   RooWorkspace wsFromJson("new");
   RooJSONFactoryWSTool newtool{wsFromJson};
   newtool.importJSONfromString(js);

   std::string const &js3 = RooJSONFactoryWSTool{wsFromJson}.exportJSONtoString();

   if (writeJsonFiles) {
      RooJSONFactoryWSTool{wsFromJson}.exportJSON(_name + "_2.json");
   }

   // Chack that JSON > WS > JSON doesn't change the JSON
   EXPECT_EQ(js, js3) << "The JSON -> WS -> JSON roundtrip did not result in the original JSON!";

   auto *newmc = dynamic_cast<RooStats::ModelConfig *>(wsFromJson.obj("ModelConfig"));
   EXPECT_TRUE(newmc != nullptr);

   RooAbsPdf *newpdf = newmc->GetPdf();
   EXPECT_TRUE(newpdf != nullptr);

   RooAbsData *data = ws->data("obsData");
   EXPECT_TRUE(data != nullptr);

   RooAbsData *newdata = wsFromJson.data("obsData");
   EXPECT_TRUE(newdata != nullptr);

   RooArgSet const &globs = *mc->GetGlobalObservables();
   RooArgSet const &globsFromJson = *newmc->GetGlobalObservables();

   setInitialFitParameters(*ws, makeModelMode);
   setInitialFitParameters(wsFromJson, makeModelMode);

   using namespace RooFit;
   using Res = std::unique_ptr<RooFitResult>;

   Res result{pdf->fitTo(*data, Strategy(1), Minos(*mc->GetParametersOfInterest()), GlobalObservables(globs),
                         PrintLevel(-1), Save())};

   Res resultFromJson{newpdf->fitTo(*newdata, Strategy(1), Minos(*newmc->GetParametersOfInterest()),
                                    GlobalObservables(globsFromJson), PrintLevel(-1), Save())};

   // Do also the reverse comparison to check that the set of constant parameters matches
   EXPECT_TRUE(result->isIdentical(*resultFromJson));
   EXPECT_TRUE(resultFromJson->isIdentical(*result));
}

/// Fit the model to data, and check parameters.
TEST_P(HFFixtureFit, Fit)
{
   const MakeModelMode makeModelMode = std::get<0>(GetParam());
   RooFit::EvalBackend evalBackend = std::get<2>(GetParam());

   constexpr bool verbose = false;

   auto simPdf = dynamic_cast<RooSimultaneous *>(ws->pdf("simPdf"));
   ASSERT_NE(simPdf, nullptr);

   auto channelPdf = dynamic_cast<RooRealSumPdf *>(ws->pdf("channel1_model"));
   ASSERT_NE(channelPdf, nullptr);

   // Test fitting the model to data
   RooAbsData *data = dynamic_cast<RooAbsData *>(ws->data("obsData"));
   ASSERT_NE(data, nullptr);

   auto mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   ASSERT_NE(mc, nullptr);

   // This tests both correct pre-caching of constant terms and (if false) that all doEval() are correct.
   for (bool constTermOptimization : {true, false}) {

      // constTermOptimization makes only sense in the legacy backend
      if (constTermOptimization && evalBackend != RooFit::EvalBackend::Legacy()) {
         continue;
      }
      SCOPED_TRACE(constTermOptimization ? "const term optimisation" : "No const term optimisation");

      // Stop if one of the previous runs had a failure to keep the terminal clean.
      if (HasFailure())
         break;

      std::unique_ptr<RooArgSet> pars(simPdf->getParameters(*data));
      // Kick parameters:
      for (auto par : *pars) {
         auto real = dynamic_cast<RooAbsRealLValue *>(par);
         if (real && !real->isConstant())
            real->setVal(real->getVal() * 0.95);
      }
      if (makeModelMode == MakeModelMode::StatSyst) {
         auto poi = dynamic_cast<RooRealVar *>(pars->find("SigXsecOverSM"));
         ASSERT_NE(poi, nullptr);
         poi->setVal(2.);
         poi->setConstant();
      }

      using namespace RooFit;
      std::unique_ptr<RooFitResult> fitResult{simPdf->fitTo(*data, evalBackend, Optimize(constTermOptimization),
                                                            GlobalObservables(*mc->GetGlobalObservables()), Save(),
                                                            PrintLevel(verbose ? 1 : -1))};
      ASSERT_NE(fitResult, nullptr);
      if (verbose)
         fitResult->Print("v");
      EXPECT_EQ(fitResult->status(), 0);

      auto checkParam = [&](const std::string &param, double target, double absPrecision = 1.e-2) {
         auto par = dynamic_cast<RooRealVar *>(fitResult->floatParsFinal().find(param.c_str()));
         if (!par) {
            // Parameter was constant in this fit
            par = dynamic_cast<RooRealVar *>(fitResult->constPars().find(param.c_str()));
            ASSERT_NE(par, nullptr) << param;
            EXPECT_DOUBLE_EQ(par->getVal(), target) << "Constant parameter " << param << " is off target.";
         } else {
            EXPECT_NEAR(par->getVal(), target, par->getError())
               << "Parameter " << param << " close to target " << target << " within uncertainty";
            EXPECT_NEAR(par->getVal(), target, absPrecision) << "Parameter " << param << " close to target " << target;
         }
      };

      if (makeModelMode == MakeModelMode::OverallSyst) {
         // Model is set up such that background scale factors should be close to 1, and signal == 2
         checkParam("SigXsecOverSM", 2.);
         checkParam("alpha_syst2", 0.);
         checkParam("alpha_syst3", 0.);
         checkParam("alpha_syst4", 0.);
         checkParam("gamma_stat_channel1_bin_0", 1.);
         checkParam("gamma_stat_channel1_bin_1", 1.);
      } else if (makeModelMode == MakeModelMode::HistoSyst) {
         // Model is set up with a -1 sigma pull on the signal shape parameter.
         checkParam("SigXsecOverSM", 2., 1.1E-1); // Higher tolerance: Expect a pull due to shape syst.
         checkParam("gamma_stat_channel1_bin_0", 1.);
         checkParam("gamma_stat_channel1_bin_1", 1.);
         checkParam("alpha_SignalShape", -0.9, 5.E-2); // Pull slightly lower than 1 because of constraint term
      } else if (makeModelMode == MakeModelMode::StatSyst) {
         // Model is set up with a -1 sigma pull on the signal shape parameter.
         checkParam("SigXsecOverSM", 2., 1.1E-1);       // Higher tolerance: Expect a pull due to shape syst.
         checkParam("gamma_stat_channel1_bin_0", 1.09); // This should be pulled
         checkParam("gamma_stat_channel1_bin_1", 1.);
      } else if (makeModelMode == MakeModelMode::ShapeSyst) {
         // This should be pulled down
         checkParam("gamma_background1Shape_bin_0", 0.8866, 0.03);
         // This should be pulled up, but not so much because the free signal
         // strength will fit the excess in this bin.
         checkParam("gamma_background2Shape_bin_1", 1.0250, 0.03);
      }
   }

   if (false) {
      auto obs = dynamic_cast<RooRealVar *>(ws->var("obs_x_channel1"));
      auto frame = obs->frame();
      data->plotOn(frame);
      channelPdf->plotOn(frame);
      channelPdf->plotOn(frame, RooFit::Components("signal_channel1_shapes"), RooFit::LineColor(kRed));
      TCanvas canv;
      frame->Draw();
      canv.Draw();
      canv.SaveAs(("HFTest" + _name + ".png").c_str());

      channelPdf->graphVizTree(("HFTest" + _name + ".dot").c_str());
   }
}

std::string getNameFromInfo(testing::TestParamInfo<HFFixture::ParamType> const &paramInfo)
{
   return getName(paramInfo.param, false);
}

INSTANTIATE_TEST_SUITE_P(
   HistFactory, HFFixture,
   testing::Combine(testing::Values(MakeModelMode::OverallSyst, MakeModelMode::HistoSyst, MakeModelMode::StatSyst,
                                    MakeModelMode::ShapeSyst),
                    testing::Values(false, true),                    // non-uniform bins or not
                    testing::Values(RooFit::EvalBackend::Cpu())), // dummy because no NLL is created
   [](testing::TestParamInfo<HFFixture::ParamType> const &paramInfo) { return getName(paramInfo.param, true); });

INSTANTIATE_TEST_SUITE_P(HistFactory, HFFixtureEval,
                         testing::Combine(testing::Values(MakeModelMode::OverallSyst, MakeModelMode::HistoSyst,
                                                          MakeModelMode::StatSyst, MakeModelMode::ShapeSyst),
                                          testing::Values(false, true), // non-uniform bins or not
                                          testing::Values(ROOFIT_EVAL_BACKENDS)),
                         getNameFromInfo);

INSTANTIATE_TEST_SUITE_P(HistFactory, HFFixtureFit,
                         testing::Combine(testing::Values(MakeModelMode::OverallSyst, MakeModelMode::HistoSyst,
                                                          MakeModelMode::StatSyst, MakeModelMode::ShapeSyst),
                                          testing::Values(false, true), // non-uniform bins or not
                                          testing::Values(ROOFIT_EVAL_BACKENDS_WITH_CODEGEN)),
                         getNameFromInfo);
