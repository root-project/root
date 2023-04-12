// Tests for the RooStats::HistFactory::JSONTool
// Authors: Carsten D. Burgard, DESY/ATLAS, 12/2021
//          Jonas Rembser, CERN 12/2022

#include <RooFitHS3/JSONIO.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooFitHS3/HistFactoryJSONTool.h>
#include <RooFit/ModelConfig.h>

#include <RooStats/HistFactory/Measurement.h>
#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>

#include <RooHelpers.h>

#include <TFile.h>
#include <TROOT.h>

#include <gtest/gtest.h>

namespace {

// If the JSON files should be written out for debugging purpose.
const bool writeJsonFiles = false;

void setupKeys()
{
   static bool isAlreadySetup = false;
   if (isAlreadySetup)
      return;

   auto etcDir = std::string(TROOT::GetEtcDir());
   RooFit::JSONIO::loadExportKeys(etcDir + "/RooFitHS3_wsexportkeys.json");
   RooFit::JSONIO::loadFactoryExpressions(etcDir + "/RooFitHS3_wsfactoryexpressions.json");

   isAlreadySetup = true;
}

class HistoWriter {
public:
   HistoWriter(int nbinsx, double xlow, double xup) : _nbinsx{nbinsx}, _xlow{xlow}, _xup{xup} {}
   void operator()(std::string const &name, std::vector<float> const &arr)
   {
      // to test code paths where name and title is treated differently
      std::string title = name + "_title";
      auto histo = std::make_unique<TH1F>(name.c_str(), title.c_str(), _nbinsx, _xlow, _xup);
      for (int i = 0; i < _nbinsx; ++i) {
         histo->SetBinContent(i + 1, arr[i]);
      }
      histo->Write();
   }

private:
   int _nbinsx = 0;
   double _xlow = 0.0;
   double _xup = 0.0;
};

void createInputFile(std::string const &inputFileName)
{
   TFile file{inputFileName.c_str(), "RECREATE"};

   HistoWriter hw{2, 1.0, 2.0};

   hw("data", {122., 112.});
   hw("signal", {20., 10.});
   hw("shapeUnc_sigDo", {15., 8.});
   hw("shapeUnc_sigUp", {29., 13.});

   hw("background1", {100., 0.});
   hw("background2", {0., 100.});
   hw("background1_statUncert", {0.05, 0.05});
}

std::unique_ptr<RooStats::HistFactory::Measurement>
measurement(const char *inputFileName = "test_hs3_histfactory_json_input.root")
{
   createInputFile(inputFileName);

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);
   gROOT->SetBatch(true);
   auto meas = std::make_unique<RooStats::HistFactory::Measurement>("meas", "meas");
   meas->SetOutputFilePrefix("./results/example_");
   meas->SetPOI("mu");
   meas->AddConstantParam("Lumi");
   meas->SetLumi(1.0);
   meas->SetLumiRelErr(0.10);
   meas->SetExportOnly(true);
   meas->SetBinHigh(2);
   // meas.AddConstantParam("syst1");
   RooStats::HistFactory::Channel chan{"channel1"};
   chan.SetData("data", inputFileName);
   chan.SetStatErrorConfig(0.01, "Poisson");

   RooStats::HistFactory::Sample sig{"signal", "signal", inputFileName};
   sig.AddOverallSys("syst1", 0.95, 1.05);
   sig.AddNormFactor("mu", 1, -3, 5);
   sig.AddHistoSys("SignalShape", "shapeUnc_sigDo", inputFileName, "", "shapeUnc_sigUp", inputFileName, "");
   chan.AddSample(sig);

   RooStats::HistFactory::Sample background1{"background1", "background1", inputFileName};
   background1.ActivateStatError("background1_statUncert", inputFileName);
   background1.AddOverallSys("syst2", 0.95, 1.05);
   chan.AddSample(background1);
   RooStats::HistFactory::Sample background2{"background2", "background2", inputFileName};
   background2.ActivateStatError();
   background2.AddOverallSys("syst3", 0.95, 1.05);
   chan.AddSample(background2);
   meas->AddChannel(chan);
   meas->CollectHistograms();
   return meas;
}

} // namespace

TEST(TestHS3HistFactoryJSON, HistFactoryJSONTool)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   setupKeys();

   std::unique_ptr<RooStats::HistFactory::Measurement> meas = measurement();
   if (writeJsonFiles) {
      RooStats::HistFactory::JSONTool{*meas}.PrintJSON("hf.json");
   }
   std::stringstream ss;
   RooStats::HistFactory::JSONTool{*meas}.PrintJSON(ss);

   std::unique_ptr<RooWorkspace> ws{RooStats::HistFactory::MakeModelAndMeasurementFast(*meas)};
   RooWorkspace wsFromJson{"ws1"};
   RooJSONFactoryWSTool{wsFromJson}.importJSONfromString(ss.str());

   auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   EXPECT_TRUE(mc != nullptr);

   auto *mcFromJson = dynamic_cast<RooStats::ModelConfig *>(wsFromJson.obj("ModelConfig"));
   EXPECT_TRUE(mcFromJson != nullptr);

   RooAbsPdf *pdf = mc->GetPdf();
   EXPECT_TRUE(pdf != nullptr);

   RooAbsPdf *pdfFromJson = mcFromJson->GetPdf();
   EXPECT_TRUE(pdfFromJson != nullptr);

   RooAbsData *data = ws->data("obsData");
   EXPECT_TRUE(data != nullptr);

   RooAbsData *dataFromJson = wsFromJson.data("obsData");
   EXPECT_TRUE(dataFromJson != nullptr);

   RooArgSet const &globs = *mc->GetGlobalObservables();

   using namespace RooFit;
   using Res = std::unique_ptr<RooFitResult>;

   Res result{pdf->fitTo(*data, Strategy(1), Minos(*mc->GetParametersOfInterest()), GlobalObservables(globs),
                         PrintLevel(-1), Save())};

   Res resultFromJson{pdfFromJson->fitTo(*dataFromJson, Strategy(1), Minos(*mcFromJson->GetParametersOfInterest()),
                                         GlobalObservablesTag("globs"), PrintLevel(-1), Save())};

   // Do also the reverse comparison to check that the set of constant parameters matches
   EXPECT_TRUE(result->isIdentical(*resultFromJson));
   EXPECT_TRUE(resultFromJson->isIdentical(*result));
}

TEST(TestHS3HistFactoryJSON, ClosureLoop)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   setupKeys();

   std::unique_ptr<RooStats::HistFactory::Measurement> meas = measurement();
   std::unique_ptr<RooWorkspace> ws{RooStats::HistFactory::MakeModelAndMeasurementFast(*meas)};

   auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   EXPECT_TRUE(mc != nullptr);

   RooAbsPdf *pdf = mc->GetPdf();
   EXPECT_TRUE(pdf != nullptr);

   std::string const &js = RooJSONFactoryWSTool{*ws}.exportJSONtoString();
   if (writeJsonFiles) {
      RooJSONFactoryWSTool{*ws}.exportJSON("hf2.json");
   }

   RooWorkspace newws("new");
   RooJSONFactoryWSTool newtool{newws};
   newtool.importJSONfromString(js);

   std::string const &js3 = RooJSONFactoryWSTool{newws}.exportJSONtoString();

   if (writeJsonFiles) {
      RooJSONFactoryWSTool{newws}.exportJSON("hf3.json");
   }

   // Chack that JSON > WS > JSON doesn't change the JSON
   EXPECT_EQ(js, js3) << "The JSON -> WS -> JSON roundtrip did not result in the original JSON!";

   auto *newmc = dynamic_cast<RooStats::ModelConfig *>(newws.obj("ModelConfig"));
   EXPECT_TRUE(newmc != nullptr);

   RooAbsPdf *newpdf = newmc->GetPdf();
   EXPECT_TRUE(newpdf != nullptr);

   RooAbsData *data = ws->data("obsData");
   EXPECT_TRUE(data != nullptr);

   RooAbsData *newdata = newws.data("obsData");
   EXPECT_TRUE(newdata != nullptr);

   RooArgSet const &globs = *mc->GetGlobalObservables();

   using namespace RooFit;
   using Res = std::unique_ptr<RooFitResult>;

   Res result{pdf->fitTo(*data, Strategy(1), Minos(*mc->GetParametersOfInterest()), GlobalObservables(globs),
                         PrintLevel(-1), Save())};

   Res resultFromJson{newpdf->fitTo(*newdata, Strategy(1), Minos(*newmc->GetParametersOfInterest()),
                                    GlobalObservablesTag("globs"), PrintLevel(-1), Save())};

   // Do also the reverse comparison to check that the set of constant parameters matches
   EXPECT_TRUE(result->isIdentical(*resultFromJson));
   EXPECT_TRUE(resultFromJson->isIdentical(*result));
}
