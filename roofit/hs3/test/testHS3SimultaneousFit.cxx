// Test for the JSON IO of a full workspace with a multi-channel model and
// data.
// Author: Jonas Rembser, CERN 02/2023

#include <RooFitHS3/JSONIO.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooGlobalFunc.h>
#include <RooProdPdf.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <RooStats/ModelConfig.h>

#include <TROOT.h>

#include <gtest/gtest.h>

namespace {

/// Generating an unbinned dataset from a binned one.
std::unique_ptr<RooDataSet> makeUnbinned(RooDataHist const &hist)
{
   RooArgSet obs(*hist.get());
   RooRealVar weight{"weight", "weight", 1.0};
   obs.add(weight, true);
   auto data = std::make_unique<RooDataSet>(hist.GetName(), hist.GetTitle(), obs, RooFit::WeightVar(weight));
   for (int i = 0; i < hist.numEntries(); ++i) {
      data->add(*hist.get(i), hist.weight(i));
   }
   return data;
}

std::unique_ptr<RooFitResult> writeJSONAndFitModel(std::string &jsonStr)
{
   using namespace RooFit;

   RooWorkspace ws{"workspace"};

   // Build two channels for different observables where the distributions
   // share one parameter: the mean for the signal.

   // Channel 1: Gaussian signal and exponential background
   ws.factory("Gaussian::sig_1(x_1[0, 10], mean[5.0, 0, 10], sigma_1[0.5, 0.1, 10.0])");
   ws.factory("Exponential::bkg_1(x_1, c_1[-0.2, -100, -0.001])");
   ws.factory("SUM::model_1(n_sig_1[10000, 0, 10000000] * sig_1, nbkg_2[100000, 0, 10000000] * bkg_1)");

   // Channel 2: Crystal ball signal and polynomial background
   ws.factory("CBShape::sig_2(x_2[0, 10], mean[5.0, 0, 10], sigma_2[0.8, 0.1, 10.0], alpha[0.9, 0.1, 10.0], "
              "ncb[1.0, 0.1, 10.0])");
   ws.factory("Polynomial::bkg_2(x_2, {a_0[3.0, -10, 10], a_1[-0.3, -10, 10], a_2[0.01, -10, 10]}, 0)");
   ws.factory("SUM::model_2(n_sig_2[30000, 0, 10000000] * sig_2, nbkg_2[100000, 0, 10000000] * bkg_2)");

   // Simultaneous PDF and model config
   ws.factory("SIMUL::simPdf(channelCat[channel_1=0, channel_2=1], channel_1=model_1, channel_2=model_2)");

   RooStats::ModelConfig modelConfig{"ModelConfig"};

   modelConfig.SetWS(ws);
   modelConfig.SetPdf("simPdf");
   modelConfig.SetParametersOfInterest("mean");
   modelConfig.SetObservables("x_1,x_2");

   ws.import(modelConfig);

   modelConfig.Print();

   RooRealVar &x1 = *ws.var("x_1");
   RooRealVar &x2 = *ws.var("x_2");
   x1.setBins(20);
   x2.setBins(20);

   RooAbsPdf &model_1 = *ws.pdf("model_1");
   RooAbsPdf &model_2 = *ws.pdf("model_2");

   std::unique_ptr<RooDataHist> data1{model_1.generateBinned(x1)};
   std::unique_ptr<RooDataHist> data2{model_2.generateBinned(x2)};

   data1->SetName("obsData_channel_1");
   data2->SetName("obsData_channel_2");

   ws.import(*data1);
   ws.import(*data2);

   RooJSONFactoryWSTool tool{ws};

   // Export before fitting to keep the prefit values
   jsonStr = tool.exportJSONtoString();

   RooRealVar weight{"weight", "weight", 1.0};
   RooDataSet obsData{"obsData",
                      "obsData",
                      {x1, x2, weight},
                      Index(*ws.cat("channelCat")),
                      Import({{"channel_1", makeUnbinned(*data1).get()}, {"channel_2", makeUnbinned(*data2).get()}}),
                      WeightVar(weight)};

   return std::unique_ptr<RooFitResult>{ws.pdf("simPdf")->fitTo(obsData, Save(), PrintLevel(-1), PrintEvalErrors(-1))};
}

std::unique_ptr<RooFitResult> readJSONAndFitModel(std::string const &jsonStr)
{
   using namespace RooFit;

   RooWorkspace ws{"workspace"};
   RooJSONFactoryWSTool tool{ws};

   tool.importJSONfromString(jsonStr);

   auto &pdf = *ws.pdf("simPdf");

   return std::unique_ptr<RooFitResult>{pdf.fitTo(*ws.data("obsData"), Save(), PrintLevel(-1), PrintEvalErrors(-1))};
}

} // namespace

TEST(RooFitHS3, SimultaneousFit)
{
   using namespace RooFit;

   auto etcDir = std::string(TROOT::GetEtcDir());
   RooFit::JSONIO::loadExportKeys(etcDir + "/RooFitHS3_wsexportkeys.json");
   RooFit::JSONIO::loadFactoryExpressions(etcDir + "/RooFitHS3_wsfactoryexpressions.json");

   std::string jsonStr;

   std::unique_ptr<RooFitResult> res1 = writeJSONAndFitModel(jsonStr);
   std::unique_ptr<RooFitResult> res2 = readJSONAndFitModel(jsonStr);

   // todo: also check the modelconfig for equality

   // The precision is not great, needs to be understood why it is not exactly the same
   EXPECT_TRUE(res2->isIdentical(*res1, 1e-3, 1e-3));
}
