// Tests for RooRealIntegral
// Authors: Jonas Rembser, CERN 10/2022

#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooHistPdf.h>
#include <RooPlot.h>
#include <RooProduct.h>
#include <RooProjectedPdf.h>
#include <RooRealIntegral.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <ROOT/StringUtils.hxx>

#include "../src/RooGenProdProj.h"

#include "gtest_wrapper.h"

#include <memory>

namespace {
RooArgList getSortedServers(RooAbsArg const &arg)
{
   // Sort alphabetically in case the two integrals didn't add the servers in
   // the same order:
   RooArgList servers{arg.servers().begin(), arg.servers().end()};
   servers.sort();
   return servers;
}
} // namespace

// Verify that the value servers of a RooRealIntegral are the direct
// mathematical value servers of the integral, and not the leaves in the
// computation graphs. For the Batch mode, it is important for the evaluation
// of the computation graph that no direct value servers are skipped. See also
// GitHub issue #11578.
TEST(RooRealIntegral, ClientServerInterface1)
{
   RooWorkspace ws;

   // This is the key in this test: the mathematically direct value server of
   // the integral is the derived "mu_mod", and not the leaf of the computation
   // graph "mu".
   ws.factory("Product::mu_mod({mu[-0.005, -5.0, 5.0], 10.0})");
   ws.factory("Gaussian::gauss(x[0, 1], mu_mod, 2.0)");

   RooRealVar &x = *ws.var("x");
   RooAbsPdf &gauss = *ws.pdf("gauss");
   RooGenericPdf pdf{"gaussWrapped", "gauss", gauss};

   std::unique_ptr<RooAbsReal> integ1{gauss.createIntegral(x, *pdf.getIntegratorConfig(), nullptr)};
   std::unique_ptr<RooAbsReal> integ2{pdf.createIntegral(x, *pdf.getIntegratorConfig(), nullptr)};

   RooArgList servers1{getSortedServers(*integ1)};
   RooArgList servers2{getSortedServers(*integ2)};

   // Check that the server structure of the Gaussian integral looks as
   // expected, which should be, if you use Print("v"):
   //
   //     (-S) RooRealVar::x ""
   //     (--) RooGenericPdf::pdf "gauss"
   //     (V-) RooProduct::mu_mod ""
   //     (V-) RooConstVar::2 "2"
   //
   // What is important is that the indirect value server "mu" doesn't appear
   // among the servers, and the direct value server "mu_mod" does.

   EXPECT_EQ(servers1.size(), 4);

   // Respect the alphabetical order here
   EXPECT_EQ(std::string(servers1[0].GetName()), "2");
   EXPECT_EQ(std::string(servers1[1].GetName()), "gauss");
   EXPECT_EQ(std::string(servers1[2].GetName()), "mu_mod");
   EXPECT_EQ(std::string(servers1[3].GetName()), "x");

   EXPECT_TRUE(servers1[0].isValueServer(*integ1));
   EXPECT_FALSE(servers1[1].isValueServer(*integ1));
   EXPECT_TRUE(servers1[2].isValueServer(*integ1));
   EXPECT_FALSE(servers1[3].isValueServer(*integ1));

   EXPECT_FALSE(servers1[0].isShapeServer(*integ1));
   EXPECT_FALSE(servers1[1].isShapeServer(*integ1));
   EXPECT_FALSE(servers1[2].isShapeServer(*integ1));
   EXPECT_TRUE(servers1[3].isShapeServer(*integ1));

   // The Gaussian PDF wrapped in a RooGenericPdf should have exactly the same
   // server structure, so let's check that:

   EXPECT_EQ(servers2.size(), servers1.size());

   for (std::size_t i = 0; i < servers1.size(); ++i) {
      RooAbsArg const &s1 = servers1[i];
      RooAbsArg const &s2 = servers2[i];

      // The 2nd server is the integrated function, which doesn't have the same
      // name (it's "gaussWrapped" for the second integral instead of "gauss")
      if (i != 1) {
         EXPECT_EQ(std::string(s1.GetName()), s2.GetName());
      }

      EXPECT_EQ(s1.isValueServer(*integ1), s2.isValueServer(*integ2));
      EXPECT_EQ(s1.isShapeServer(*integ1), s2.isShapeServer(*integ2));
   }
}

/// Here we are integrating a function that has shape servers to verify that
/// they are correctly propagated as shape servers to the integral.
TEST(RooRealIntegral, IntegrateFuncWithShapeServers)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumIntegration, true};

   RooWorkspace ws;
   ws.factory("Product::mu_mod({mu[-0.005, -5.0, 5.0], 10.0})");
   ws.factory("Gaussian::gauss(x[0, 1], mu_mod, sigma[1, 0.5, 2.0])");

   RooRealVar &x = *ws.var("x");
   RooAbsReal &muMod = *ws.function("mu_mod");
   RooRealVar &sigma = *ws.var("sigma");
   RooAbsPdf &gauss = *ws.pdf("gauss");
   RooGenericPdf pdf("pdf", "gauss", gauss);

   // Project over sigma, meaning sigma should now become a shape server
   RooProjectedPdf gaussProj("gaussProj", "", gauss, sigma);

   EXPECT_TRUE(x.isValueServer(gaussProj));
   EXPECT_FALSE(x.isShapeServer(gaussProj));
   EXPECT_TRUE(muMod.isValueServer(gaussProj));
   EXPECT_FALSE(muMod.isShapeServer(gaussProj));
   EXPECT_FALSE(sigma.isValueServer(gaussProj));
   EXPECT_TRUE(sigma.isShapeServer(gaussProj));

   // Integrating also over x, so both x and sigma should now be shape servers of the integral
   std::unique_ptr<RooAbsReal> integ1{gaussProj.createIntegral(x, *pdf.getIntegratorConfig(), nullptr)};

   EXPECT_FALSE(x.isValueServer(*integ1)); // x is now not a value server anymore
   EXPECT_TRUE(x.isShapeServer(*integ1));
   EXPECT_TRUE(muMod.isValueServer(*integ1));
   EXPECT_FALSE(muMod.isShapeServer(*integ1));
   EXPECT_FALSE(sigma.isValueServer(*integ1));
   EXPECT_TRUE(sigma.isShapeServer(*integ1)); // sigma should still be shape server!

   // Also check that the number of servers is right (should be 3 for x,
   // mu, and sigma, and 1 more for the underlying PDF)
   EXPECT_EQ(gaussProj.servers().size(), 4);
   EXPECT_EQ(integ1->servers().size(), 4);
}

// Verify that using observable clones -- i.e., variables with the same names
// as the ones in the computation graph -- does not change the client-server
// structure of a RooRealIntegral. Covers GitHub issue #11637.
TEST(RooRealIntegral, UseCloneAsIntegrationVariable1)
{
   RooRealVar x1{"x", "x1", -10, 10};
   RooRealVar x2{"x", "x2", -10, 10};

   RooGenericPdf gauss{"gauss", "std::exp(-0.5 * (x*x))", x1};

   RooRealIntegral integ1{"integ1", "", gauss, x1};
   RooRealIntegral integ2{"integ2", "", gauss, x2};

   // Check that client-server structure is as expected.
   for (auto const &integ : {integ1, integ2}) {

      RooArgList servers{getSortedServers(integ)};

      EXPECT_EQ(std::string(servers[0].GetName()), "gauss");
      EXPECT_EQ(std::string(servers[1].GetName()), "x");

      EXPECT_FALSE(servers[0].isValueServer(integ));
      EXPECT_FALSE(servers[1].isValueServer(integ));

      EXPECT_FALSE(servers[0].isShapeServer(integ));
      EXPECT_TRUE(servers[1].isShapeServer(integ));

      EXPECT_EQ(servers.size(), 2);
   }
}

// More testing of observable clones as integration variables. This time
// hitting the more general case where the algorithm also needs to find clients
// of the original variable correctly ("xShifted" in this test).
TEST(RooRealIntegral, UseCloneAsIntegrationVariable2)
{
   RooRealVar x1{"x", "x1", 0.0, 10};
   RooRealVar x2{"x", "x2", 0.0, 10};

   RooRealVar shift("shift", "", 0, -10, 10);
   RooFormulaVar xShifted("x_shifted", "x + shift", {x1, shift});

   RooDataHist dataHist("dataHist", "", x1);
   RooHistPdf func("func", "", xShifted, x1, dataHist);

   RooRealIntegral integ1{"integ1", "", func, x1};
   RooRealIntegral integ2{"integ2", "", func, x2};

   // Check that client-server structure is as expected.
   for (auto const &integ : {integ1, integ2}) {

      RooArgList servers{getSortedServers(integ)};

      EXPECT_EQ(std::string(servers[0].GetName()), "func");
      EXPECT_EQ(std::string(servers[1].GetName()), "shift");
      EXPECT_EQ(std::string(servers[2].GetName()), "x");

      EXPECT_FALSE(servers[0].isValueServer(integ));
      EXPECT_TRUE(servers[1].isValueServer(integ));
      EXPECT_FALSE(servers[2].isValueServer(integ));

      EXPECT_FALSE(servers[0].isShapeServer(integ));
      EXPECT_FALSE(servers[1].isShapeServer(integ));
      EXPECT_TRUE(servers[2].isShapeServer(integ));

      EXPECT_EQ(servers.size(), 3);
   }
}

/// Make sure that the normalization set for a RooAddPdf is always defined when
/// numerically integrating a RooProdPdf where the RooAddPdf is one of the
/// factors. Covers GitHub #11476 and JIRA issue ROOT-9436.
///
/// Disabled for now because the fix to the bug that is covered by this unit
/// test caused a severe performance problem and was reverted. The performance
/// regression is covered by another unit test in this file, called
/// "ProjectConditional".
TEST(RooRealIntegral, DISABLED_Issue11476)
{
   // Silence the info about numeric integration because we don't care about it
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumIntegration, true};

   RooWorkspace ws{"ws"};
   ws.factory("Gaussian::gs(x[0,10], mu[2, 0, 10], sg[2, 0.1, 10])");
   ws.factory("Exponential::ex(x, lm[-0.1, -10, 0])");
   ws.factory("SUM::gs_ex(f[0.5, 0, 1] * gs, ex)");
   ws.factory("Gaussian::gs_1(x, mu_1[4, 0, 10], sg_1[2, 0.1, 10])");
   ws.factory("PROD::pdf(gs_1, gs_ex)");

   RooRealVar &x = *ws.var("x");
   RooAbsPdf &pdf = *ws.pdf("pdf");

   // Store the logged warnings for missing normalization sets
   RooHelpers::HijackMessageStream hijack(RooFit::WARNING, RooFit::Eval);

   std::unique_ptr<RooDataSet> data{pdf.generate(x, 10000)};

   // Check that there were no warnings (covers GitHub issue #11476)
   const std::string messages = hijack.str();
   std::cout << messages;
   EXPECT_TRUE(messages.empty()) << "Unexpected warnings were issued! Stream contents: " << hijack.str();

   // Verify that plot is correctly normalized
   std::unique_ptr<RooPlot> frame{x.frame()};
   data->plotOn(frame.get());
   pdf.plotOn(frame.get());

   // If the normalization of the PDF in the plot is correct, the chi-square
   // will be low. Covers JIRA issue ROOT-9436.
   EXPECT_LE(frame->chiSquare(), 1.0)
      << "The chi-square of the plot is too high, the normalization of the PDF is probably wrong!";
}

class LevelTest : public testing::TestWithParam<int> {
   void SetUp() override { _level = GetParam(); }

protected:
   int _level;

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

/// Related to GitHub issue #11814.
TEST_P(LevelTest, ProjectConditional)
{
   RooHelpers::HijackMessageStream hijack(RooFit::INFO, RooFit::NumIntegration);

   constexpr bool verbose = false;

   RooWorkspace w;
   w.factory("Poisson::px(x[150,0,500],sum::splusb(s[0,0,100],b[100,0,300]))");
   w.factory("Poisson::py(y[100,0,500],prod::taub(tau[1.],b))");
   w.factory("Uniform::prior_b(b)");

   RooRealVar &x = *w.var("x");
   RooRealVar &b = *w.var("b");

   std::unique_ptr<RooAbsReal> function;
   std::size_t expectedNumInts = 0;

   // The following three code blocks all cover the issue, but generate the
   // computation graph using different levels of interacting with RooFit, from low to high level.
   switch (_level) {

   case 1: {
      // Version 1) Manually reproducing the RooGenProdProj instance that the
      //            RooProdPdf would create under the hood:
      RooAbsPdf &px = *w.pdf("px");
      RooAbsPdf &py = *w.pdf("py");
      RooArgSet iset{b};
      RooArgSet eset{};
      std::unique_ptr<RooAbsReal> pxNormed{px.createIntegral({}, x)};
      auto genProdProj = std::make_unique<RooGenProdProj>("genProdProj", "", RooArgSet{py}, eset, iset, nullptr);
      auto prod = std::make_unique<RooProduct>("prod", "", RooArgList{*genProdProj, *pxNormed});
      function = std::unique_ptr<RooAbsReal>{prod->createIntegral(iset)};

      function->addOwnedComponents(std::move(pxNormed));
      function->addOwnedComponents(std::move(genProdProj));
      function->addOwnedComponents(std::move(prod));

      expectedNumInts = 2;
   } break;

   case 2: {
      // Version 2) Doing the final projection integral manually:
      w.factory("PROD::foo(px|b,py,prior_b)");
      function = std::unique_ptr<RooAbsReal>{w.pdf("foo")->createIntegral({b}, {b, x})};
      expectedNumInts = 3;
   } break;

   case 3: {
      // Version 3) High-level Projection in RooWorkspace factory language, as
      //            it originally appeared in the RooStats tutorials:
      w.factory("PROJ::averagedModel(PROD::foo(px|b,py,prior_b),b)");
      function = std::unique_ptr<RooAbsReal>{static_cast<RooAbsReal *>(w.pdf("averagedModel")->Clone())};
      expectedNumInts = 4;
   } break;

   default: break;
   }

   RooArgSet nset{b, x};

   for (int i = 0; i < 10; ++i) {
      x.setVal(i % 500);

      const double val = function->getVal(nset);
      if (verbose) {
         std::cout << val << std::endl;
      }
   }

   EXPECT_LE(ROOT::Split(hijack.str(), "\n", true).size(), expectedNumInts)
      << "More numeric integrals than expected! This might be okay, but could also point to a performance regression "
         "for the model covered in this unit test. Please investigate, and increase the number of expected numeric "
         "integrals in this test if they are not related to performance regressions.";
}

INSTANTIATE_TEST_SUITE_P(RooRealIntegral, LevelTest, testing::Values(1, 2, 3),
                         [](testing::TestParamInfo<LevelTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "Level" << paramInfo.param;
                            return ss.str();
                         });

// If we integrate a model that uses RooLinearVar and should be able to get
// integrated analytically, this should also work if we integrate over variable
// clones because RooFit considers them identical. Covers GitHub issue #12646.
TEST(RooRealIntegral, RooLinearVarModelIntegratedOverVariableClones)
{
   RooWorkspace ws;
   ws.factory("LinearVar::x2(x[0, 1], 1, 0)");
   ws.factory("LinearVar::y2(y[0, 1], 1, 0)");

   // RooGaussian can integrate over x or mu, but not both still, the issue is
   // visible regardless
   ws.factory("Gaussian::gauss(x2, y2, 0.2)");

   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");
   RooAbsPdf &gauss = *ws.pdf("gauss");

   // There should be no numeric integration happening
   std::unique_ptr<RooAbsReal> integral{gauss.createIntegral({y}, {x, y})};
   EXPECT_TRUE(static_cast<RooRealIntegral &>(*integral).numIntRealVars().empty());

   RooRealVar xCopy{x};
   RooRealVar yCopy{y};

   // Also if we use clones of the observables, it should not make a difference
   std::unique_ptr<RooAbsReal> integral2{gauss.createIntegral({yCopy}, {xCopy, yCopy})};
   EXPECT_TRUE(static_cast<RooRealIntegral &>(*integral2).numIntRealVars().empty());
}

// Make sure that RooFit realizes that Gaussian(x, mu, sigma(x)) needs to be
// integrated analytically.
// Covers GitHub issue #14320.
TEST(RooRealIntegral, GaussianWithSigmaDependingOnX)
{
   RooWorkspace ws;
   ws.factory("x[100., 1000.]");
   ws.factory("expr::res('0.07 * x + 2.0',x)");
   ws.factory("Gaussian::sig(x, 200., res)");

   auto &x = *ws.var("x");
   auto &model = *ws.pdf("sig");

   std::unique_ptr<RooAbsReal> integ1{model.createIntegral(x)};
   double val1 = integ1->getVal();

   // Force numerical integration for the reference value.
   model.forceNumInt(true);
   std::unique_ptr<RooAbsReal> integ2{model.createIntegral(x)};
   double val2 = integ2->getVal();

   EXPECT_EQ(val1, val2);
}
