// Tests for the HistFactory
// Authors: Stephan Hageboeck, CERN  01/2019
//          Jonas Rembser, CERN  06/2023

#include <RooStats/HistFactory/Measurement.h>
#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>
#ifdef HISTFACTORY_XML
#include <RooStats/HistFactory/ConfigParser.h>
#endif
#include <RooFit/ModelConfig.h>

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
#include <TH1D.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <gtest/gtest.h>

#include "../../roofitcore/test/gtest_wrapper.h"
#include "HFFixture.h"

namespace {

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
   const bool useBatchMode = std::get<2>(GetParam()) != RooFit::EvalBackend(RooFit::EvalBackend::Value::Legacy);

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
   std::unique_ptr<RooFitResult> fitResult{simPdf->fitTo(
      *data, evalBackend, GlobalObservables(*mc->GetGlobalObservables()), Save(), PrintLevel(verbose ? 1 : -1))};
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
                    testing::Values(false, true),                 // non-uniform bins or not
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

// Issue #20697: Sample::AddShapeFactor() now allows to set the initial value
// and the range of the ShapeFactor gammas (just like AddNormFactor() does for
// the NormFactors). This is important e.g. for ABCD estimates, where the
// hard-coded default range can cause convergence problems.
//
// These settings need to survive being persisted, so this test checks that the
// value and range make it through:
//   1. a ROOT file round trip (Measurement::writeToFile),
//   2. an XML file round trip (Measurement::PrintXML / ConfigParser), and
//   3. into the actual gamma parameters of the generated workspace.
TEST(HistFactory, ShapeFactorValueAndRange)
{
   using namespace RooStats::HistFactory;
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   // Deliberately use non-default values and a range that differs from the
   // hard-coded default of [0, 1000].
   const double sfVal = 2.0;
   const double sfLow = 0.1;
   const double sfHigh = 12.0;

   const std::string inputFileName = "TestShapeFactorRange_input.root";
   {
      TFile f(inputFileName.c_str(), "RECREATE");
      auto *data = new TH1D("data", "data", 2, 1, 2);
      auto *signal = new TH1D("signal", "signal", 2, 1, 2);
      auto *bkg = new TH1D("background", "background", 2, 1, 2);
      data->SetBinContent(1, 220);
      data->SetBinContent(2, 230);
      signal->SetBinContent(1, 10);
      signal->SetBinContent(2, 20);
      bkg->SetBinContent(1, 200);
      bkg->SetBinContent(2, 200);
      for (auto *h : {data, signal, bkg})
         f.WriteTObject(h);
   }

   auto makeMeasurement = [&]() {
      Measurement meas("meas", "meas");
      meas.SetOutputFilePrefix("TestShapeFactorRange");
      meas.SetPOI("SigXsecOverSM");
      meas.AddConstantParam("Lumi");
      meas.SetLumi(1.0);
      meas.SetLumiRelErr(0.10);

      Channel chan("channel1");
      chan.SetData("data", inputFileName);

      Sample sig("signal", "signal", inputFileName);
      sig.AddNormFactor("SigXsecOverSM", 1, 0, 3);
      chan.AddSample(sig);

      // The new overload under test: ShapeFactor with custom value and range.
      Sample bkg("background", "background", inputFileName);
      bkg.AddShapeFactor("bkgShape", sfVal, sfLow, sfHigh);
      chan.AddSample(bkg);

      meas.AddChannel(chan);
      meas.CollectHistograms();
      return meas;
   };

   // Fetch the (single) ShapeFactor stored in a measurement.
   auto getShapeFactor = [](Measurement &meas) -> ShapeFactor & {
      Channel &chan = meas.GetChannel("channel1");
      for (Sample &sample : chan.GetSamples()) {
         if (!sample.GetShapeFactorList().empty())
            return sample.GetShapeFactorList().front();
      }
      throw std::runtime_error("ShapeFactor not found in measurement");
   };

   auto checkShapeFactor = [&](Measurement &meas, const char *context) {
      ShapeFactor &sf = getShapeFactor(meas);
      EXPECT_DOUBLE_EQ(sf.GetVal(), sfVal) << context;
      EXPECT_DOUBLE_EQ(sf.GetLow(), sfLow) << context;
      EXPECT_DOUBLE_EQ(sf.GetHigh(), sfHigh) << context;
   };

   // 0. Sanity check on the in-memory measurement.
   {
      Measurement meas = makeMeasurement();
      checkShapeFactor(meas, "in-memory measurement");
   }

   // 1. ROOT file round trip.
   {
      Measurement meas = makeMeasurement();
      const std::string rootFileName = "TestShapeFactorRange_meas.root";
      {
         TFile outFile(rootFileName.c_str(), "RECREATE");
         meas.writeToFile(&outFile);
      }
      TFile inFile(rootFileName.c_str(), "READ");
      std::unique_ptr<Measurement> measFromFile{inFile.Get<Measurement>("meas")};
      ASSERT_NE(measFromFile, nullptr);
      checkShapeFactor(*measFromFile, "ROOT file round trip");
   }

   // 2. XML file round trip.
   //    The ConfigParser is only built with the xml build option, so this part
   //    of the test is skipped when ROOT is built without XML support.
#ifdef HISTFACTORY_XML
   {
      Measurement meas = makeMeasurement();
      const std::string xmlDir = "TestShapeFactorRangeXML";
      meas.PrintXML(xmlDir);

      // The generated XML files refer to the DTD by relative path, so it has to
      // be available next to them for the validating parser to find it.
      gSystem->CopyFile(TString::Format("%s/HistFactorySchema.dtd", TROOT::GetEtcDir().Data()),
                        TString::Format("%s/HistFactorySchema.dtd", xmlDir.c_str()), true);

      ConfigParser parser;
      std::vector<Measurement> measFromXML = parser.GetMeasurementsFromXML(xmlDir + "/meas.xml");
      ASSERT_EQ(measFromXML.size(), 1u);
      checkShapeFactor(measFromXML.front(), "XML file round trip");
   }
#endif

   // 3. End to end: the gamma parameters of the workspace pick up the requested
   //    value and range.
   {
      Measurement meas = makeMeasurement();
      std::unique_ptr<RooWorkspace> ws{MakeModelAndMeasurementFast(meas)};
      ASSERT_NE(ws, nullptr);
      for (const char *name : {"gamma_bkgShape_bin_0", "gamma_bkgShape_bin_1"}) {
         auto *gamma = ws->var(name);
         ASSERT_NE(gamma, nullptr) << name;
         EXPECT_DOUBLE_EQ(gamma->getVal(), sfVal) << name;
         EXPECT_DOUBLE_EQ(gamma->getMin(), sfLow) << name;
         EXPECT_DOUBLE_EQ(gamma->getMax(), sfHigh) << name;
      }
   }
}
