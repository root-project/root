// Tests for the RooStats::HistFactory::JSONTool
// Authors: Carsten D. Burgard, DESY/ATLAS, 12/2021
//          Jonas Rembser, CERN 12/2022

#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooFitHS3/HistFactoryJSONTool.h>

#include <RooStats/ModelConfig.h>
#include <RooStats/HistFactory/Measurement.h>
#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>

#include <RooHelpers.h>

#include <TROOT.h>

#include <gtest/gtest.h>

namespace {

void toJSON(RooStats::HistFactory::Measurement &meas, std::string const &fname)
{
   RooStats::HistFactory::JSONTool tool{meas};
   tool.PrintJSON(fname);
}

std::unique_ptr<RooWorkspace> toWS(RooStats::HistFactory::Measurement &meas)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);
   return std::unique_ptr<RooWorkspace>{RooStats::HistFactory::MakeModelAndMeasurementFast(meas)};
}

std::unique_ptr<RooWorkspace> importToWS(std::string const &infile, std::string const &wsname)
{
   auto ws = std::make_unique<RooWorkspace>(wsname.c_str());
   RooJSONFactoryWSTool tool{*ws};
   tool.importJSON(infile);
   return ws;
}

std::unique_ptr<RooStats::HistFactory::Measurement>
measurement(std::string const &inputFileName = "test_hs3_histfactory_json_input.root")
{
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
   chan.SetData("data", inputFileName.c_str());
   chan.SetStatErrorConfig(0.01, "Poisson");
   RooStats::HistFactory::Sample sig{"signal", "signal", inputFileName.c_str()};
   sig.AddOverallSys("syst1", 0.95, 1.05);
   sig.AddNormFactor("mu", 1, -3, 5);
   chan.AddSample(sig);
   RooStats::HistFactory::Sample background1{"background1", "background1", inputFileName.c_str()};
   background1.ActivateStatError("background1_statUncert", inputFileName);
   background1.AddOverallSys("syst2", 0.95, 1.05);
   chan.AddSample(background1);
   RooStats::HistFactory::Sample background2{"background2", "background2", inputFileName.c_str()};
   background2.ActivateStatError();
   background2.AddOverallSys("syst3", 0.95, 1.05);
   chan.AddSample(background2);
   meas->AddChannel(chan);
   meas->CollectHistograms();
   return meas;
}

} // namespace

TEST(TestHS3HistFactoryJSON, Create)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   toJSON(*measurement(), "hf.json");
}

TEST(TestHS3HistFactoryJSON, Closure)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   std::unique_ptr<RooStats::HistFactory::Measurement> meas = measurement();
   toJSON(*meas, "hf.json");
   std::unique_ptr<RooWorkspace> ws = toWS(*meas);
   std::unique_ptr<RooWorkspace> wsFromJson = importToWS("hf.json", "ws1");

   auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   EXPECT_TRUE(mc != nullptr);

   auto *mcFromJson = dynamic_cast<RooStats::ModelConfig *>(wsFromJson->obj("ModelConfig"));
   EXPECT_TRUE(mcFromJson != nullptr);

   RooAbsPdf *pdf = mc->GetPdf();
   EXPECT_TRUE(pdf != nullptr);

   RooAbsPdf *pdfFromJson = wsFromJson->pdf(meas->GetName());
   EXPECT_TRUE(pdfFromJson != nullptr);

   RooAbsData *data = ws->data("obsData");
   EXPECT_TRUE(data != nullptr);

   RooAbsData *dataFromJson = wsFromJson->data("obsData");
   EXPECT_TRUE(dataFromJson != nullptr);

   using namespace RooFit;

   pdf->fitTo(*data, Strategy(1), Minos(*mc->GetParametersOfInterest()), GlobalObservables(*mc->GetGlobalObservables()),
              PrintLevel(-1));

   pdfFromJson->fitTo(*dataFromJson, Strategy(1), Minos(*mcFromJson->GetParametersOfInterest()),
                      GlobalObservables(*mcFromJson->GetGlobalObservables()), PrintLevel(-1));

   const double muVal = ws->var("mu")->getVal();
   const double muJsonVal = wsFromJson->var("mu")->getVal();

   EXPECT_NEAR(muJsonVal, muVal, 1e-4);         // absolute tolerance
   EXPECT_NEAR(muJsonVal, muVal, 1e-4 * muVal); // relative tolerance
}

TEST(TestHS3HistFactoryJSON, ClosureLoop)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   std::unique_ptr<RooStats::HistFactory::Measurement> meas = measurement();
   std::unique_ptr<RooWorkspace> ws = toWS(*meas);

   RooJSONFactoryWSTool tool{*ws};
   std::string const &js = tool.exportJSONtoString();

   RooWorkspace newws("new");
   RooJSONFactoryWSTool newtool{newws};
   newtool.importJSONfromString(js);

   auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   EXPECT_TRUE(mc != nullptr);

   auto *newmc = dynamic_cast<RooStats::ModelConfig *>(newws.obj("ModelConfig"));
   EXPECT_TRUE(newmc != nullptr);

   RooAbsPdf *pdf = mc->GetPdf();
   EXPECT_TRUE(pdf != nullptr);

   RooAbsPdf *newpdf = newmc->GetPdf();
   EXPECT_TRUE(newpdf != nullptr);

   RooAbsData *data = ws->data("obsData");
   EXPECT_TRUE(data != nullptr);

   RooAbsData *newdata = newws.data("obsData");
   EXPECT_TRUE(newdata != nullptr);

   using namespace RooFit;

   pdf->fitTo(*data, Strategy(1), Minos(*mc->GetParametersOfInterest()), GlobalObservables(*mc->GetGlobalObservables()),
              PrintLevel(-1));

   newpdf->fitTo(*newdata, Strategy(1), Minos(*newmc->GetParametersOfInterest()),
                 GlobalObservables(*newmc->GetGlobalObservables()), PrintLevel(-1));

   const double muVal = ws->var("mu")->getVal();
   const double muNewVal = newws.var("mu")->getVal();

   EXPECT_NEAR(muNewVal, muVal, 1e-4);         // absolute tolerance
   EXPECT_NEAR(muNewVal, muVal, 1e-4 * muVal); // relative tolerance
}
