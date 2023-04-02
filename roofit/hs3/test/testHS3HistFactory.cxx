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

void createInputFile(std::string const &inputFileName)
{
   TH1F data("data", "data", 2, 1.0, 2.0);

   TH1F signal("signal", "signal histogram (pb)", 2, 1.0, 2.0);
   TH1F systUncDo("shapeUnc_sigDo", "signal shape uncert.", 2, 1.0, 2.0);
   TH1F systUncUp("shapeUnc_sigUp", "signal shape uncert.", 2, 1.0, 2.0);

   TH1F background1("background1", "background 1 histogram (pb)", 2, 1.0, 2.0);
   TH1F background2("background2", "background 2 histogram (pb)", 2, 1.0, 2.0);
   TH1F background1_statUncert("background1_statUncert", "statUncert", 2, 1.0, 2.0);

   data.SetBinContent(1, 122.);
   data.SetBinContent(2, 112.);

   signal.SetBinContent(1, 20.);
   signal.SetBinContent(2, 10.);

   systUncDo.SetBinContent(1, 15.);
   systUncDo.SetBinContent(2, 8.);

   systUncUp.SetBinContent(1, 29.);
   systUncUp.SetBinContent(2, 13.);

   background1.SetBinContent(1, 100.);
   background1.SetBinContent(2, 0.);

   background2.SetBinContent(1, 0.);
   background2.SetBinContent(2, 100.);

   background1_statUncert.SetBinContent(1, 0.05);
   background1_statUncert.SetBinContent(2, 0.05);

   TFile file{inputFileName.c_str(), "RECREATE"};

   data.Write();
   signal.Write();
   systUncDo.Write();
   systUncUp.Write();
   background1.Write();
   background2.Write();
   background1_statUncert.Write();
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

   // TODO: At this point, the global observables are not written into the JSON
   // by the HistFactoryJSONTool. This should be fixed. For now, we mirror the
   // global observables from the reference workspace.
   RooArgSet const &globs = *mc->GetGlobalObservables();
   RooArgSet globsInJson;
   for (RooAbsArg const *arg : globs) {
      globsInJson.add(*wsFromJson.var(arg->GetName()));
   };

   using namespace RooFit;
   using Res = std::unique_ptr<RooFitResult>;

   Res result{pdf->fitTo(*data, Strategy(1), Minos(*mc->GetParametersOfInterest()), GlobalObservables(globs),
                         PrintLevel(-1), Save())};

   Res resultFromJson{pdfFromJson->fitTo(*dataFromJson, Strategy(1), Minos(*mcFromJson->GetParametersOfInterest()),
                                         GlobalObservables(globsInJson), PrintLevel(-1), Save())};

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

   // TODO: At this point, the global observables are not written into the JSON
   // by the HistFactoryJSONTool. This should be fixed. For now, we mirror the
   // global observables from the reference workspace.
   RooArgSet const &globs = *mc->GetGlobalObservables();
   RooArgSet globsInJson;
   for (RooAbsArg const *arg : globs) {
      globsInJson.add(*newws.var(arg->GetName()));
   };

   using namespace RooFit;
   using Res = std::unique_ptr<RooFitResult>;

   Res result{pdf->fitTo(*data, Strategy(1), Minos(*mc->GetParametersOfInterest()), GlobalObservables(globs),
                         PrintLevel(-1), Save())};

   Res resultFromJson{newpdf->fitTo(*newdata, Strategy(1), Minos(*newmc->GetParametersOfInterest()),
                                    GlobalObservables(globsInJson), PrintLevel(-1), Save())};

   // Do also the reverse comparison to check that the set of constant parameters matches
   EXPECT_TRUE(result->isIdentical(*resultFromJson));
   EXPECT_TRUE(resultFromJson->isIdentical(*result));
}
