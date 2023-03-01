// Tests for the RooJSONFactoryWSTool
// Authors: Carsten D. Burgard, DESY/ATLAS, 12/2021
//          Jonas Rembser, CERN 12/2022

#include <RooFitHS3/JSONIO.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooRealVar.h>
#include <RooConstVar.h>
#include <RooWorkspace.h>
#include <RooGlobalFunc.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooProdPdf.h>
#include <RooCategory.h>

#include <TROOT.h>

#include <gtest/gtest.h>

namespace {

// Validate the JSON IO for a given RooAbsReal in a RooWorkspace. The workspace
// will be written out and read back, and then the values of the old and new
// RooAbsReal will be compared for equality in each bin of the observable that
// is called "x" by convention.
int validate(RooWorkspace &ws1, std::string const &argName)
{
   RooWorkspace ws2;

   auto etcDir = std::string(TROOT::GetEtcDir());
   RooFit::JSONIO::loadExportKeys(etcDir + "/RooFitHS3_wsexportkeys.json");
   RooFit::JSONIO::loadFactoryExpressions(etcDir + "/RooFitHS3_wsfactoryexpressions.json");

   RooJSONFactoryWSTool tool1{ws1};
   RooJSONFactoryWSTool tool2{ws2};

   tool2.importJSONfromString(tool1.exportJSONtoString());

   RooRealVar &x1 = *ws1.var("x");
   RooRealVar &x2 = *ws2.var("x");

   RooAbsReal &arg1 = *ws1.function(argName);
   RooAbsReal &arg2 = *ws2.function(argName);

   RooArgSet nset1{x1};
   RooArgSet nset2{x2};

   bool allGood = true;
   for (int i = 0; i < x1.numBins(); ++i) {
      x1.setBin(i);
      x2.setBin(i);
      const double val1 = arg1.getVal(nset1);
      const double val2 = arg2.getVal(nset2);
      allGood &= val1 == val2;
   }

   return allGood ? 0 : 1;
}

int validate(std::vector<std::string> const &expressions)
{
   RooWorkspace ws;
   for (std::size_t iExpr = 0; iExpr < expressions.size() - 1; ++iExpr) {
      ws.factory(expressions[iExpr]);
   }
   const std::string argName = ws.factory(expressions.back())->GetName();
   return validate(ws, argName);
}

int validate(RooAbsArg const &arg)
{
   RooWorkspace ws;
   ws.import(arg, RooFit::Silence());
   return validate(ws, arg.GetName());
}

} // namespace

TEST(RooFitHS3, RooAddPdf)
{
   int status =
      validate({"Gaussian::signalModel(x[5.20, 5.30], sigmean[5.28, 5.20, 5.30], sigwidth[0.0027, 0.001, 1.])",
                "ArgusBG::background(x, 5.291, argpar[-20.0, -100., -1.])",
                "SUM::model(nsig[200, 0., 10000] * signalModel, nbkg[800, 0., 10000] * background)"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooArgusBG)
{
   int status = validate({"ArgusBG::argusBG(x[0, 20], x0[10], c[-1], p[0.5])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooBifurGauss)
{
   int status = validate({"BifurGauss::bifurGauss(x[0, 10], mean[5], sigmaL[1.0, 0.1, 10], sigmaR[2.0, 0.1, 10])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooCBShape)
{
   int status = validate({"CBShape::cbShape(x[-10, 10], x0[0], sigma[2.0], alpha[1.4], n[1.2])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooGaussian)
{
   int status = validate({"Gaussian::gaussian(x[0, 10], mean[5], sigma[1.0, 0.1, 10])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooHistPdf)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x{"x", "x", 0.0, 0.02};
   x.setBins(2);

   RooDataHist dataHist{"myDataHist", "myDataHist", x};
   dataHist.set(0, 25.0, 5.0);
   dataHist.set(1, 25.0, 5.0);

   int status = validate(RooHistPdf{"histPdf", "histPdf", x, dataHist});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooPoisson)
{
   int status = validate({"Poisson::poisson(x[0, 10], mean[5])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooPolynomial)
{
   // Test different values for "lowestOrder"
   int status = 0;
   status = validate({"Polynomial::poly0(x[0, 10], {a_0[3.0], a_1[-0.3, -10, 10], a_2[0.01, -10, 10]}, 0)"});
   EXPECT_EQ(status, 0);
   status = validate({"Polynomial::poly1(x[0, 10], {a_1[-0.1, -10, 10], a_2[0.003, -10, 10]}, 1)"});
   EXPECT_EQ(status, 0);
   status = validate({"Polynomial::poly1(x[0, 10], {a_2[0.003, -10, 10]}, 2)"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, SimultaneousGaussians)
{
   using namespace RooFit;

   // Import keys and factory expressions files for the RooJSONFactoryWSTool.
   auto etcDir = std::string(TROOT::GetEtcDir());
   RooFit::JSONIO::loadExportKeys(etcDir + "/RooFitHS3_wsexportkeys.json");
   RooFit::JSONIO::loadFactoryExpressions(etcDir + "/RooFitHS3_wsfactoryexpressions.json");

   // Create a test model: RooSimultaneous with Gaussian in one component, and
   // product of two Gaussians in the other.
   RooRealVar x("x", "x", -8, 8);
   RooRealVar mean("mean", "mean", 0, -8, 8);
   RooRealVar sigma("sigma", "sigma", 0.3, 0.1, 10);
   RooGaussian g1("g1", "g1", x, mean, sigma);
   RooGaussian g2("g2", "g2", x, mean, RooConst(0.3));
   RooProdPdf model("model", "model", RooArgList{g1, g2});
   RooGaussian model_ctl("model_ctl", "model_ctl", x, mean, sigma);
   RooCategory sample("sample", "sample", {{"physics", 0}, {"control", 1}});
   RooSimultaneous simPdf("simPdf", "simultaneous pdf", sample);
   simPdf.addPdf(model, "physics");
   simPdf.addPdf(model_ctl, "control");

   // this is a handy way of triggering the creation of a ModelConfig upon re-import
   simPdf.setAttribute("toplevel");

   // Export to JSON
   {
      RooWorkspace ws{"workspace"};
      ws.import(simPdf, RooFit::Silence());
      RooJSONFactoryWSTool tool{ws};
      tool.exportJSON("simPdf.json");
      // Output can be pretty-printed with `python -m json.tool simPdf.json`
   }

   // Import JSON
   {
      RooWorkspace ws{"workspace"};
      RooJSONFactoryWSTool tool{ws};
      tool.importJSON("simPdf.json");

      ASSERT_TRUE(ws.pdf("g1"));
      ASSERT_TRUE(ws.pdf("g2"));
   }
}
