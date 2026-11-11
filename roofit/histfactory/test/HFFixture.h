// Toy HistFactory model, shared by the HistFactory and the HS3 tests.

#ifndef ROOFIT_HISTFACTORY_TEST_HFFIXTURE_H
#define ROOFIT_HISTFACTORY_TEST_HFFIXTURE_H

#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>
#include <RooStats/HistFactory/Measurement.h>

#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooWorkspace.h>

#include <TFile.h>
#include <TH1D.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

/// What kind of model is set up. Use this to instantiate
/// a test suite.
enum class MakeModelMode {
   OverallSyst,
   HistoSyst,
   StatSyst,
   ShapeSyst
};

using HFTestParam = std::tuple<MakeModelMode, bool, RooFit::EvalBackend>;

inline std::string getName(HFTestParam const &param, bool ignoreBackend = false)
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

#endif
