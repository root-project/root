// Tests for the HistFactory
// Authors: Stephan Hageboeck, CERN  01/2019

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/MakeModelAndMeasurementsFast.h"
#include "RooStats/HistFactory/Sample.h"
#include "RooStats/ModelConfig.h"

#include "RooWorkspace.h"
#include "RooArgSet.h"
#include "RooSimultaneous.h"
#include "RooRealSumPdf.h"
#include "RooRealVar.h"
#include "RooHelpers.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "RunContext.h"

#include "TROOT.h"
#include "TFile.h"
#include "TCanvas.h"
#include "gtest/gtest.h"

#include <set>

using namespace RooStats;
using namespace RooStats::HistFactory;

TEST(Sample, CopyAssignment)
{
  Sample s("s");
  {
    Sample s1("s1");
    auto hist1 = new TH1D("hist1", "hist1", 10, 0, 10);
    s1.SetHisto(hist1);
    s = s1;
    //Now go out of scope. Should delete hist1, that's owned by s1.
  }
  
  auto hist = s.GetHisto();
  ASSERT_EQ(hist->GetNbinsX(), 10);
}


TEST(HistFactory, Read_ROOT6_16_Model) {
  std::string filename = "./ref_6.16_example_UsingC_channel1_meas_model.root";
  std::unique_ptr<TFile> file(TFile::Open(filename.c_str()));
  if (!file || !file->IsOpen()) {
    filename = TROOT::GetRootSys() + "/roofit/histfactory/test/" + filename;
    file.reset(TFile::Open(filename.c_str()));
  }

  ASSERT_TRUE(file && file->IsOpen());
  RooWorkspace* ws;
  file->GetObject("channel1", ws);
  ASSERT_NE(ws, nullptr);

  auto mc = dynamic_cast<RooStats::ModelConfig*>(ws->obj("ModelConfig"));
  ASSERT_NE(mc, nullptr);

  RooAbsPdf* pdf = mc->GetPdf();
  ASSERT_NE(pdf, nullptr);

  const RooArgSet* obs = mc->GetObservables();
  ASSERT_NE(obs, nullptr);

  EXPECT_NEAR(pdf->getVal(), 0.17488817, 1.E-8);
  EXPECT_NEAR(pdf->getVal(*obs), 0.95652174, 1.E-8);
}


TEST(HistFactory, Read_ROOT6_16_Combined_Model) {
  std::string filename = "./ref_6.16_example_UsingC_combined_meas_model.root";
  std::unique_ptr<TFile> file(TFile::Open(filename.c_str()));
  if (!file || !file->IsOpen()) {
    filename = TROOT::GetRootSys() + "/roofit/histfactory/test/" + filename;
    file.reset(TFile::Open(filename.c_str()));
  }

  ASSERT_TRUE(file && file->IsOpen());
  RooWorkspace* ws;
  file->GetObject("combined", ws);
  ASSERT_NE(ws, nullptr);

  auto mc = dynamic_cast<RooStats::ModelConfig*>(ws->obj("ModelConfig"));
  ASSERT_NE(mc, nullptr);

  RooAbsPdf* pdf = mc->GetPdf();
  ASSERT_NE(pdf, nullptr);

  const RooArgSet* obs = mc->GetObservables();
  ASSERT_NE(obs, nullptr);

  EXPECT_NEAR(pdf->getVal(), 0.17488817, 1.E-8);
  EXPECT_NEAR(pdf->getVal(*obs), 0.95652174, 1.E-8);
}


/// What kind of model is set up. Use this to instantiate
/// a test suite.
/// \note Make sure that equidistant bins have even numbers,
/// so those tests can be found using `% 2 == kEquidistantBins`.
enum MakeModelMode {kEquidistantBins=0, kCustomBins=1,
  kEquidistantBins_histoSyst = 2, kCustomBins_histoSyst = 3,
  kEquidistantBins_statSyst  = 4, kCustomBins_statSyst  = 5};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Fixture class to set up a toy hist factory model.
/// In the SetUp() phase
/// - A file with histograms is created. Depending on the value of MakeModelMode,
/// equidistant or custom bins are used, and shape systematics are added.
/// - A Measurement with the histograms in the file is created.
/// - The corresponding workspace is created.
class HFFixture : public testing::TestWithParam<MakeModelMode> {
public:
  std::string _inputFile{"TestMakeModel.root"};
  static constexpr bool _verbose = false;
  double _customBins[3] = {0., 1.8, 2.};
  const double _targetMu = 2.;
  const double _targetNominal[2] = {110., 120.};
  const double _targetSysUp[2]   = {112., 140.};
  const double _targetSysDo[2]   = {108., 100.};
  std::unique_ptr<RooWorkspace> ws;
  std::set<std::string> _systNames; // Systematics defined during set up

  void SetUp() {
    {
      TFile example(_inputFile.c_str(), "RECREATE");
      TH1F *data, *signal, *bkg1, *bkg2, *statUnc, *systUncUp, *systUncDo;
      data = signal = bkg1 = bkg2 = statUnc = systUncUp = systUncDo = nullptr;
      if (GetParam() % 2 == kEquidistantBins) {
        data = new TH1F("data","data", 2,1,2);
        signal = new TH1F("signal","signal histogram (pb)", 2,1,2);
        bkg1 = new TH1F("background1","background 1 histogram (pb)", 2,1,2);
        bkg2 = new TH1F("background2","background 2 histogram (pb)", 2,1,2);
        statUnc = new TH1F("background1_statUncert", "statUncert", 2,1,2);
        systUncUp = new TH1F("shapeUnc_sigUp", "signal shape uncert.", 2,1,2);
        systUncDo = new TH1F("shapeUnc_sigDo", "signal shape uncert.", 2,1,2);
      } else if (GetParam() % 2 == kCustomBins) {
        data = new TH1F("data","data", 2, _customBins);
        signal = new TH1F("signal","signal histogram (pb)", 2, _customBins);
        bkg1 = new TH1F("background1","background 1 histogram (pb)", 2, _customBins);
        bkg2 = new TH1F("background2","background 2 histogram (pb)", 2, _customBins);
        statUnc = new TH1F("background1_statUncert", "statUncert", 2, _customBins);
        systUncUp = new TH1F("shapeUnc_sigUp", "signal shape uncert.", 2, _customBins);
        systUncDo = new TH1F("shapeUnc_sigDo", "signal shape uncert.", 2, _customBins);
      } else {
        // Harden the test and make clang-tidy happy:
        FAIL() << "This should not be reachable.";
      }

      bkg1->SetBinContent(1, 100);
      bkg2->SetBinContent(2, 100);

      for (unsigned int bin=0; bin < 2; ++bin) {
        signal->   SetBinContent(bin+1, _targetNominal[bin] - 100.);
        systUncUp->SetBinContent(bin+1, _targetSysUp[bin]   - 100.);
        systUncDo->SetBinContent(bin+1, _targetSysDo[bin]   - 100.);

        if (GetParam() <= kCustomBins) {
          data->SetBinContent(bin+1, _targetMu * signal->GetBinContent(bin+1) + 100.);
        } else if (GetParam() <= kCustomBins_histoSyst) {
          // Set data such that alpha = -1., fit should pull parameter.
          data->SetBinContent(bin+1, _targetMu * systUncDo->GetBinContent(bin+1) + 100.);
        } else if (GetParam() <= kCustomBins_statSyst) {
          // Tighten the stat. errors of the model, and kick bin 0, so the gammas have to adapt
          signal->SetBinError(bin+1, 0.1 * sqrt(signal->GetBinContent(bin+1)));
          bkg1  ->SetBinError(bin+1, 0.1 * sqrt(bkg1  ->GetBinContent(bin+1)));
          bkg2  ->SetBinError(bin+1, 0.1 * sqrt(bkg2  ->GetBinContent(bin+1)));

          data->SetBinContent(bin+1, _targetMu * signal->GetBinContent(bin+1) + 100. + (bin == 0 ? 50. : 0.));
        }

        // A small statistical uncertainty
        statUnc->SetBinContent(bin+1, .05);  // 5% uncertainty
      }


      for (auto hist : {data, signal, bkg1, bkg2, statUnc, systUncUp, systUncDo}) {
        example.WriteTObject(hist);
      }
    }


    // Create the measurement
    Measurement meas("meas", "meas");

    meas.SetOutputFilePrefix( "example_variableBins" );
    meas.SetPOI( "SigXsecOverSM" );
    meas.AddConstantParam("alpha_syst1");
    meas.AddConstantParam("Lumi");
    if (GetParam() == kEquidistantBins_histoSyst || GetParam() == kCustomBins_histoSyst) {
      // We are testing the shape systematics. Switch off the normalisation
      // systematics for the background here:
      meas.AddConstantParam("alpha_syst2");
      meas.AddConstantParam("alpha_syst4");
      meas.AddConstantParam("gamma_stat_channel1_bin_0");
      meas.AddConstantParam("gamma_stat_channel1_bin_1");
    } else if (GetParam() == kEquidistantBins_statSyst || GetParam() == kCustomBins_statSyst) {
      // Fix all systematics but the gamma parameters
      // Cannot set the POI constant here, happens in the fit test.
      meas.AddConstantParam("alpha_syst2");
      meas.AddConstantParam("alpha_syst3");
      meas.AddConstantParam("alpha_syst4");
      meas.AddConstantParam("alpha_SignalShape");
    }

    meas.SetExportOnly(true);

    meas.SetLumi( 1.0 );
    meas.SetLumiRelErr( 0.10 );


    // Create a channel
    Channel chan( "channel1" );
    chan.SetData( "data", _inputFile);
    chan.SetStatErrorConfig( 0.005, "Poisson" );
    _systNames.insert("gamma_stat_channel1_bin_0");
    _systNames.insert("gamma_stat_channel1_bin_1");



    // Now, create some samples

    // Create the signal sample
    Sample signal( "signal", "signal", _inputFile);
    signal.AddOverallSys( "syst1",  0.95, 1.05 );
    _systNames.insert("alpha_syst1");

    signal.AddNormFactor( "SigXsecOverSM", 1, 0, 3 );
    if (GetParam() >= kEquidistantBins_histoSyst) {
      signal.AddHistoSys("SignalShape",
          "shapeUnc_sigDo", _inputFile, "",
          "shapeUnc_sigUp", _inputFile, "");
      _systNames.insert("alpha_SignalShape");
    }
    chan.AddSample( signal );

    // Background 1
    Sample background1( "background1", "background1", _inputFile);
    background1.ActivateStatError( "background1_statUncert", _inputFile);
    background1.AddOverallSys( "syst2", 0.95, 1.05  );
    background1.AddOverallSys( "syst3", 0.99, 1.01  );
    _systNames.insert("alpha_syst2");
    _systNames.insert("alpha_syst3");
    chan.AddSample( background1 );

    // Background 2
    Sample background2( "background2", "background2", _inputFile);
    background2.ActivateStatError();
    background2.AddOverallSys( "syst3", 0.99, 1.01  );
    background2.AddOverallSys( "syst4", 0.95, 1.05  );
    _systNames.insert("alpha_syst3");
    _systNames.insert("alpha_syst4");
    chan.AddSample( background2 );


    // Done with this channel
    // Add it to the measurement:
    meas.AddChannel( chan );

    if (!_verbose) {
      RooMsgService::instance().getStream(1).minLevel = RooFit::PROGRESS;
      RooMsgService::instance().getStream(2).minLevel = RooFit::WARNING;
    }
    RooHelpers::HijackMessageStream hijackW(RooFit::WARNING, RooFit::HistFactory);

    // Collect the histograms from their files,
    meas.CollectHistograms();

    // Now, create the measurement
    ws.reset( MakeModelAndMeasurementFast(meas) );

    EXPECT_TRUE(hijackW.str().empty()) << "Warnings logged for HistFactory:\n" << hijackW.str();
  }

  void TearDown() {

  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Test that the model consists of what is expected
TEST_P(HFFixture, ModelProperties) {
  auto simPdf = dynamic_cast<RooSimultaneous*>(ws->pdf("simPdf"));
  ASSERT_NE(simPdf, nullptr);

  auto channelPdf = dynamic_cast<RooRealSumPdf*>(ws->pdf("channel1_model"));
  ASSERT_NE(channelPdf, nullptr);

  auto obs = dynamic_cast<RooRealVar*>(ws->var("obs_x_channel1"));

  // Nice to inspect the model if needed:
  if (false)
    channelPdf->graphVizTree("/tmp/graphVizTree.dot");

  // Check bin widths
  ASSERT_NE(obs, nullptr);
  if (GetParam() % 2 == kEquidistantBins) {
    EXPECT_DOUBLE_EQ(obs->getBinWidth(0), 0.5);
    EXPECT_DOUBLE_EQ(obs->getBinWidth(1), 0.5);
    EXPECT_EQ(obs->numBins(), 2);
  } else if (GetParam() % 2 == kCustomBins) {
    EXPECT_DOUBLE_EQ(obs->getBinWidth(0), _customBins[1] - _customBins[0]);
    EXPECT_DOUBLE_EQ(obs->getBinWidth(1), _customBins[2] - _customBins[1]);
    EXPECT_EQ(obs->numBins(), 2);
  }

  RooStats::ModelConfig* mc = dynamic_cast<RooStats::ModelConfig*>(ws->obj("ModelConfig"));
  ASSERT_NE(mc, nullptr);


  // Check that parameters are in the model
  for (const auto& systName : _systNames) {
    auto& var = *ws->var(systName.c_str());

    EXPECT_TRUE(channelPdf->dependsOnValue(var)) << "Expect channel pdf to depend on " << systName;
    if (!var.isConstant()) {
      EXPECT_NE(mc->GetNuisanceParameters()->find(systName.c_str()), nullptr) << systName << " should be in list of nuisance parameters.";
    }
  }

  // Check that sub models depend on their systematic uncertainties.
  for (auto& subModelName : std::initializer_list<std::string>{"signal_channel1_shapes", "background1_channel1_shapes", "background2_channel1_shapes"}) {
    auto subModel = ws->function(subModelName.c_str());
    ASSERT_NE(subModel, nullptr) << "Unable to retrieve sub model with name " << subModelName;
    if (subModelName.find("signal") != std::string::npos) {
      EXPECT_FALSE(subModel->dependsOn(*ws->var("gamma_stat_channel1_bin_0")));
      EXPECT_FALSE(subModel->dependsOn(*ws->var("gamma_stat_channel1_bin_1")));
    } else {
        EXPECT_TRUE(subModel->dependsOn(*ws->var("gamma_stat_channel1_bin_0")));
        EXPECT_TRUE(subModel->dependsOn(*ws->var("gamma_stat_channel1_bin_1")));
    }
  }

  EXPECT_TRUE (ws->function("signal_channel1_scaleFactors")->dependsOn(*ws->var("SigXsecOverSM")));
  EXPECT_TRUE (ws->function("signal_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst1")));
  EXPECT_FALSE(ws->function("signal_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst2")));
  EXPECT_FALSE(ws->function("signal_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst3")));
  EXPECT_FALSE(ws->function("signal_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst4")));

  EXPECT_FALSE(ws->function("background1_channel1_scaleFactors")->dependsOn(*ws->var("SigXsecOverSM")));
  EXPECT_FALSE(ws->function("background1_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst1")));
  EXPECT_TRUE (ws->function("background1_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst2")));
  EXPECT_TRUE (ws->function("background1_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst3")));
  EXPECT_FALSE(ws->function("background1_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst4")));

  EXPECT_FALSE(ws->function("background2_channel1_scaleFactors")->dependsOn(*ws->var("SigXsecOverSM")));
  EXPECT_FALSE(ws->function("background2_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst1")));
  EXPECT_FALSE(ws->function("background2_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst2")));
  EXPECT_TRUE (ws->function("background2_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst3")));
  EXPECT_TRUE (ws->function("background2_channel1_scaleFactors")->dependsOn(*ws->var("alpha_syst4")));

  EXPECT_EQ(*mc->GetParametersOfInterest()->begin(), ws->var("SigXsecOverSM"));
}


/// Test that the values returned are as expected.
TEST_P(HFFixture, Evaluation) {
  auto simPdf = dynamic_cast<RooSimultaneous*>(ws->pdf("simPdf"));
  ASSERT_NE(simPdf, nullptr);

  auto channelPdf = dynamic_cast<RooRealSumPdf*>(ws->pdf("channel1_model"));
  ASSERT_NE(channelPdf, nullptr);

  auto obs = dynamic_cast<RooRealVar*>(ws->var("obs_x_channel1"));

  RooStats::ModelConfig* mc = dynamic_cast<RooStats::ModelConfig*>(ws->obj("ModelConfig"));
  ASSERT_NE(mc, nullptr);

  // Test evaluating the model:
  double normResults[2] = {0., 0.};
  for (unsigned int i=0; i < 2; ++i) {
    obs->setBin(i);
    EXPECT_NEAR(channelPdf->getVal(),
        _targetNominal[i]/obs->getBinWidth(i), 1.E-9);
    EXPECT_NEAR(channelPdf->getVal(mc->GetObservables()),
        _targetNominal[i]/obs->getBinWidth(i)/(_targetNominal[0]+_targetNominal[1]), 1.E-9);
    normResults[i] = channelPdf->getVal(mc->GetObservables());
  }
  EXPECT_NEAR(normResults[0] * obs->getBinWidth(0) + normResults[1] * obs->getBinWidth(1), 1, 1.E-9) << "Integral over PDF range should be 1.";


  // Test that shape uncertainties have an effect:
  if (GetParam() >= kEquidistantBins_histoSyst) {
    auto var = ws->var("alpha_SignalShape");
    ASSERT_NE(var, nullptr);

    // Test syst up:
    var->setVal(1.);
    for (unsigned int i=0; i < 2; ++i) {
      obs->setBin(i);
      EXPECT_NEAR(channelPdf->getVal(),
          _targetSysUp[i]/obs->getBinWidth(i), 1.E-6);
      EXPECT_NEAR(channelPdf->getVal(mc->GetObservables()),
          _targetSysUp[i]/obs->getBinWidth(i)/(_targetSysUp[0]+_targetSysUp[1]), 1.E-6);
    }

    // Test syst down:
    var->setVal(-1.);
    for (unsigned int i=0; i < 2; ++i) {
      obs->setBin(i);
      EXPECT_NEAR(channelPdf->getVal(),
          _targetSysDo[i]/obs->getBinWidth(i), 1.E-6);
      EXPECT_NEAR(channelPdf->getVal(mc->GetObservables()),
          _targetSysDo[i]/obs->getBinWidth(i)/(_targetSysDo[0]+_targetSysDo[1]), 1.E-6);
    }
  }
}


/// Test that the values returned are as expected.
TEST_P(HFFixture, BatchEvaluation) {
  RooHelpers::HijackMessageStream evalMessages(RooFit::INFO, RooFit::FastEvaluations);

  auto simPdf = dynamic_cast<RooSimultaneous*>(ws->pdf("simPdf"));
  ASSERT_NE(simPdf, nullptr);

  auto channelPdf = dynamic_cast<RooRealSumPdf*>(ws->pdf("channel1_model"));
  ASSERT_NE(channelPdf, nullptr);

  auto obs = dynamic_cast<RooRealVar*>(ws->var("obs_x_channel1"));

  RooStats::ModelConfig* mc = dynamic_cast<RooStats::ModelConfig*>(ws->obj("ModelConfig"));
  ASSERT_NE(mc, nullptr);

  // Test evaluating the model:
  rbc::RunContext evalDataOrig;
  auto batch = evalDataOrig.makeBatch(obs, 2);
  obs->setBin(0);
  batch[0] = obs->getVal();
  obs->setBin(1);
  batch[1] = obs->getVal();

  rbc::RunContext evalData1;
  evalData1.spans = evalDataOrig.spans;
  auto results = channelPdf->getValues(evalData1, nullptr);
  rbc::RunContext evalData2;
  evalData2.spans = evalDataOrig.spans;
  auto normResults = channelPdf->getValues(evalData2, mc->GetObservables());

  for (unsigned int i=0; i < 2; ++i) {
    obs->setBin(i);
    EXPECT_NEAR(results[i],
        _targetNominal[i]/obs->getBinWidth(i), 1.E-9);
    EXPECT_NEAR(normResults[i],
        _targetNominal[i]/obs->getBinWidth(i)/(_targetNominal[0]+_targetNominal[1]), 1.E-9);
  }
  EXPECT_NEAR(normResults[0] * obs->getBinWidth(0) + normResults[1] * obs->getBinWidth(1), 1, 1.E-9) << "Integral over PDF range should be 1.";


  // Test that shape uncertainties have an effect:
  if (GetParam() >= kEquidistantBins_histoSyst) {
    auto var = ws->var("alpha_SignalShape");
    ASSERT_NE(var, nullptr);

    // Test syst up:
    var->setVal(1.);
    rbc::RunContext evalData3;
    evalData3.spans = evalDataOrig.spans;
    auto resultsSyst = channelPdf->getValues(evalData3, nullptr);
    rbc::RunContext evalData4;
    evalData4.spans = evalDataOrig.spans;
    auto normResultsSyst = channelPdf->getValues(evalData4, mc->GetObservables());
    for (unsigned int i=0; i < 2; ++i) {
      EXPECT_NEAR(resultsSyst[i],
          _targetSysUp[i]/obs->getBinWidth(i), 1.E-6);
      EXPECT_NEAR(normResultsSyst[i],
          _targetSysUp[i]/obs->getBinWidth(i)/(_targetSysUp[0]+_targetSysUp[1]), 1.E-6);
    }

    // Test syst down:
    var->setVal(-1.);
    rbc::RunContext evalData5;
    evalData5.spans = evalDataOrig.spans;
    resultsSyst = channelPdf->getValues(evalData5, nullptr);
    rbc::RunContext evalData6;
    evalData6.spans = evalDataOrig.spans;
    normResultsSyst = channelPdf->getValues(evalData6, mc->GetObservables());
    for (unsigned int i=0; i < 2; ++i) {
      obs->setBin(i);
      EXPECT_NEAR(resultsSyst[i],
          _targetSysDo[i]/obs->getBinWidth(i), 1.E-6);
      EXPECT_NEAR(normResultsSyst[i],
          _targetSysDo[i]/obs->getBinWidth(i)/(_targetSysDo[0]+_targetSysDo[1]), 1.E-6);
    }
  }

  EXPECT_TRUE(evalMessages.str().empty()) << "RooFit issued " << evalMessages.str().substr(0, 1000) << " [...]";
}


/// Fit the model to data, and check parameters.
TEST_P(HFFixture, Fit) {
  constexpr bool createPlot = false;
  constexpr bool verbose = false;

  auto simPdf = dynamic_cast<RooSimultaneous*>(ws->pdf("simPdf"));
  ASSERT_NE(simPdf, nullptr);

  auto channelPdf = dynamic_cast<RooRealSumPdf*>(ws->pdf("channel1_model"));
  ASSERT_NE(channelPdf, nullptr);

  // Test fitting the model to data
  RooAbsData* data = dynamic_cast<RooAbsData*>(ws->data("obsData"));
  ASSERT_NE(data, nullptr);

  RooStats::ModelConfig* mc = dynamic_cast<RooStats::ModelConfig*>(ws->obj("ModelConfig"));
  ASSERT_NE(mc, nullptr);

  for (rbc::BatchMode batchFit : {rbc::Cpu, rbc::Off}) {
    for (bool constTermOptimisation : {true, false}) { // This tests both correct pre-caching of constant terms and (if false) that all evaluateSpan() are correct.
      SCOPED_TRACE(batchFit ? "Batch fit" : "Normal fit");
      SCOPED_TRACE(constTermOptimisation ? "const term optimisation" : "No const term optimisation");

      // Stop if one of the previous runs had a failure to keep the terminal clean.
      if (HasFailure())
        break;

      std::unique_ptr<RooArgSet> pars( simPdf->getParameters(*data) );
      // Kick parameters:
      for (auto par : *pars) {
        auto real = dynamic_cast<RooAbsRealLValue*>(par);
        if (real && !real->isConstant())
          real->setVal(real->getVal() * 0.95);
      }
      if (GetParam() >= kEquidistantBins_statSyst) {
        auto poi = dynamic_cast<RooRealVar*>(pars->find("SigXsecOverSM"));
        ASSERT_NE(poi, nullptr);
        poi->setVal(2.);
        poi->setConstant();
      }

      auto fitResult = simPdf->fitTo(*data,
          RooFit::BatchMode(batchFit),
          RooFit::Optimize(constTermOptimisation),
          RooFit::GlobalObservables(*mc->GetGlobalObservables()),
          RooFit::Save(),
          RooFit::PrintLevel(verbose ? 1 : -1));
      ASSERT_NE(fitResult, nullptr);
      if (verbose)
        fitResult->Print();
      EXPECT_EQ(fitResult->status(), 0);


      auto checkParam = [fitResult](const std::string& param, double target, double absPrecision) {
        auto par = dynamic_cast<RooRealVar*>(fitResult->floatParsFinal().find(param.c_str()));
        if (!par) {
          // Parameter was constant in this fit
          par = dynamic_cast<RooRealVar*>(fitResult->constPars().find(param.c_str()));
          ASSERT_NE(par, nullptr);
          EXPECT_DOUBLE_EQ(par->getVal(), target) << "Constant parameter " << param << " is off target.";
        } else {
          EXPECT_NEAR(par->getVal(), target, par->getError()) << "Parameter " << param << " close to target " << target << " within uncertainty";
          EXPECT_NEAR(par->getVal(), target, absPrecision) << "Parameter " << param << " close to target " << target;
        }
      };

      if (GetParam() <= kCustomBins) {
        // Model is set up such that background scale factors should be close to 1, and signal == 2
        checkParam("SigXsecOverSM", 2., 1.E-2);
        checkParam("alpha_syst2", 0., 1.E-2);
        checkParam("alpha_syst3", 0., 1.E-2);
        checkParam("alpha_syst4", 0., 1.E-2);
        checkParam("gamma_stat_channel1_bin_0", 1., 1.E-2);
        checkParam("gamma_stat_channel1_bin_1", 1., 1.E-2);
      } else if (GetParam() <= kCustomBins_histoSyst) {
        // Model is set up with a -1 sigma pull on the signal shape parameter.
        checkParam("SigXsecOverSM", 2., 1.E-1); // Higher tolerance: Expect a pull due to shape syst.
        checkParam("alpha_syst2", 0., 1.E-2);
        checkParam("alpha_syst3", 0., 3.E-2); // Micro pull due to shape syst.
        checkParam("alpha_syst4", 0., 1.E-2);
        checkParam("gamma_stat_channel1_bin_0", 1., 1.E-2);
        checkParam("gamma_stat_channel1_bin_1", 1., 1.E-2);
        checkParam("alpha_SignalShape", -0.9, 5.E-2); // Pull slightly lower than 1 because of constraint term
      } else if (GetParam() <= kCustomBins_statSyst) {
        // Model is set up with a -1 sigma pull on the signal shape parameter.
        checkParam("SigXsecOverSM", 2., 1.E-1); // Higher tolerance: Expect a pull due to shape syst.
        checkParam("alpha_syst2", 0., 1.E-2);
        checkParam("alpha_syst3", 0., 1.E-2);
        checkParam("alpha_syst4", 0., 1.E-2);
        checkParam("gamma_stat_channel1_bin_0", 1.09, 1.E-2); // This should be pulled
        checkParam("gamma_stat_channel1_bin_1", 1., 1.E-2);
        checkParam("alpha_SignalShape", 0., 1.E-2);
      }
    }
  }


  if (createPlot) {
    auto obs = dynamic_cast<RooRealVar*>(ws->var("obs_x_channel1"));
    auto frame = obs->frame();
    data->plotOn(frame);
    channelPdf->plotOn(frame);
    channelPdf->plotOn(frame, RooFit::Components("signal_channel1_shapes"), RooFit::LineColor(kRed));
    TCanvas canv;
    frame->Draw();
    canv.Draw();
    canv.SaveAs(("/tmp/HFTest" + std::to_string(GetParam()) + ".png").c_str());

    channelPdf->graphVizTree(("/tmp/HFTest" + std::to_string(GetParam()) + ".dot").c_str());
  }
}


INSTANTIATE_TEST_SUITE_P(MakeModelWithNormSysts,
    HFFixture,
    testing::Values(kEquidistantBins, kCustomBins));

INSTANTIATE_TEST_SUITE_P(MakeModelWithHistoSysts,
    HFFixture,
    testing::Values(kEquidistantBins_histoSyst, kCustomBins_histoSyst));

INSTANTIATE_TEST_SUITE_P(MakeModelWithHistoAndStatSysts,
    HFFixture,
    testing::Values(kEquidistantBins_statSyst, kCustomBins_statSyst));

