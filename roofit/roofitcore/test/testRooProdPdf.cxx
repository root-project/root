// Tests for the RooProdPdf
// Authors: Stephan Hageboeck, CERN  02/2019
//          Jonas Rembser, CERN, June 2021

#include <RooAddPdf.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooGenericPdf.h>
#include <RooProdPdf.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>
#include <RooHelpers.h>

#include <Math/PdfFuncMathCore.h>

#include "gtest_wrapper.h"

#include <memory>
#include <sstream>
#include <string>

class TestProdPdf : public ::testing::TestWithParam<std::tuple<int, RooFit::EvalBackend>> {
public:
   TestProdPdf() : _evalBackend{RooFit::EvalBackend::Legacy()} {}

private:
   void SetUp() override
   {
      RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

      datap = std::unique_ptr<RooDataSet>{prod.generate(x, 1000)};
      a.setConstant(true);

      _optimize = std::get<0>(GetParam());
      _evalBackend = std::get<1>(GetParam());
   }

protected:
   constexpr static double bTruth = -0.5;

   RooRealVar x{"x", "x", 2., 0., 5.};
   RooRealVar a{"a", "a", -0.2, -5., 0.};
   RooRealVar b{"b", "b", bTruth, -5., 0.};

   RooGenericPdf c1{"c1", "exp(x[0]*x[1])", {x, a}};
   RooGenericPdf c2{"c2", "exp(x[0]*x[1])", {x, b}};
   RooProdPdf prod{"mypdf", "mypdf", {c1, c2}};
   std::unique_ptr<RooDataSet> datap;

   int _optimize = 0;
   RooFit::EvalBackend _evalBackend;
};

TEST_P(TestProdPdf, CachingOpt)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

   using namespace RooFit;
   prod.fitTo(*datap, Optimize(_optimize), PrintLevel(-1), _evalBackend);
   EXPECT_LT(std::abs(b.getVal() - bTruth), b.getError() * 2.5) // 2.5-sigma compatibility check
      << "b=" << b.getVal() << " +- " << b.getError() << " doesn't match truth value with O" << _optimize << ".";
}

INSTANTIATE_TEST_SUITE_P(RooProdPdf, TestProdPdf,
                         testing::Combine(testing::Values(0, 1, 2),
                                          testing::Values(ROOFIT_EVAL_BACKENDS)),
                         [](testing::TestParamInfo<TestProdPdf::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "opt" << std::get<0>(paramInfo.param) << std::get<1>(paramInfo.param).name();
                            return ss.str();
                         });

TEST(RooProdPdf, TestGetPartIntList)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl1{RooFit::ERROR, 0u, RooFit::InputArguments, true};
   RooHelpers::LocalChangeMsgLevel chmsglvl2{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

   // This test checks if RooProdPdf::getPartIntList factorizes the integrals
   // as expected, for the example of a three dimensional RooProdPdf.

   RooWorkspace ws;

   double a = 10.;
   double b = 4.;
   double c = 2.5;

   auto &x = static_cast<RooRealVar &>(*ws.factory("x[0, 0, " + std::to_string(a) + "]"));
   auto &y = static_cast<RooRealVar &>(*ws.factory("y[0, 0, " + std::to_string(b) + "]"));
   auto &z = static_cast<RooRealVar &>(*ws.factory("z[0, 0, " + std::to_string(c) + "]"));

   // Factorize the product in one 1D and one 2D pdf to get a more complicated
   // and complete test case.
   ws.factory("Uniform::pdf1({x})");
   ws.factory("Uniform::pdf2({y, z})");

   // Product of all the pdfs.
   auto &prod = static_cast<RooProdPdf &>(*ws.factory("PROD::prod(pdf1, pdf2)"));

   RooArgSet normSetNada{};
   RooArgSet normSetX{x};
   RooArgSet normSetY{y};
   RooArgSet normSetZ{z};
   RooArgSet normSetXY{x, y};
   RooArgSet normSetXZ{x, z};
   RooArgSet normSetYZ{y, z};
   RooArgSet normSetXYZ{x, y, z};

   EXPECT_DOUBLE_EQ(prod.getVal(normSetNada), 1.0);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetX), 1. / a);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetY), 1. / b);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetZ), 1. / c);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetXY), 1. / a / b);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetXZ), 1. / a / c);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetYZ), 1. / b / c);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetXYZ), 1. / a / b / c);
}

TEST(RooProdPdf, TestDepsAreCond)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::Minimization, true};

   using namespace RooFit;

   RooWorkspace ws;

   ws.factory("GaussModel::gm(x[0, 1], 0.0, xErr[0.0001, 0.1])");
   ws.factory("Decay::decayPdf(x, tau[0.4, 0, 1], gm, RooDecay::SingleSided)");
   ws.factory("Gamma::errPdf(xErr, 4.0, 0.005, 0.0)");

   RooRealVar &x = *ws.var("x");
   RooRealVar &xErr = *ws.var("xErr");
   RooRealVar &tau = *ws.var("tau");
   RooAbsPdf &decayPdf = *ws.pdf("decayPdf");
   RooAbsPdf &errPdf = *ws.pdf("errPdf");

   // What we want: decayPdf(x|xErr)*errPdf(xErr):
   RooProdPdf pdf1("pdf1", "", {errPdf}, Conditional(decayPdf, x, false));

   // Should be the same as pdf1:
   RooProdPdf pdf2("pdf2", "", {errPdf}, Conditional(decayPdf, xErr, true));

   std::unique_ptr<RooDataSet> data{pdf1.generate({x, xErr}, NumEvents(10000))};

   auto resetParameters = [&]() {
      tau.setVal(0.4);
      tau.setError(0.0);
   };

   using ResultPtr = std::unique_ptr<RooFitResult>;

   resetParameters();
   ResultPtr result2{pdf1.fitTo(*data, Save(), EvalBackend::Cpu(), PrintLevel(-1))};
   resetParameters();
   ResultPtr result4{pdf2.fitTo(*data, Save(), EvalBackend::Cpu(), PrintLevel(-1))};

   EXPECT_TRUE(result4->isIdentical(*result2)) << "alternative model fit is inconsistent!";

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
   resetParameters();
   ResultPtr result1{pdf1.fitTo(*data, Save(), EvalBackend::Legacy(), PrintLevel(-1))};
   resetParameters();
   ResultPtr result3{pdf2.fitTo(*data, Save(), EvalBackend::Legacy(), PrintLevel(-1))};

   EXPECT_TRUE(result2->isIdentical(*result1)) << "legacy fit is inconsistent!";
   EXPECT_TRUE(result4->isIdentical(*result1)) << "alternative model legacy fit is inconsistent!";
#endif
}

/// This test covers a potential problem with the custom normalization ranges
/// of PDFs: the value of a RooProdPdf should not depend on the normalization
/// ranges of its clients. TODO: this is still a problem! It needs to be fixed,
/// and then the test can be enabled and this note should be removed.
TEST(RooProdPdf, DISABLED_ChangeServerNormSetForProdPdfInAddPdf)
{
   // Define the model
   RooWorkspace ws;
   ws.factory("PROD::sig(Gaussian(x[-10, 10], 1.0, 1.0), Gaussian(y[-10, 10], 1.0, 1.0))");
   ws.factory("PROD::bkg(Polynomial(x, {}), Polynomial(y, {}))");
   ws.factory("SUM::model(f[0.5, 0, 1] * sig, bkg)");
   ws.factory("SUM::modelFixed(f * sig, bkg)");
   ws.factory("EXPR::modelRef('f * 3.0/2000. * x * x + (1 - f) / 20.', {x, f})");

   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");

   RooAddPdf &model = static_cast<RooAddPdf &>(*ws.pdf("model"));

   // Define signal and sideband regions
   x.setRange("SB1", -10, +10);
   y.setRange("SB1", -10, 0);

   x.setRange("SB2", -10, 0);
   y.setRange("SB2", 0, +10);

   x.setRange("SIG", 0, +10);
   y.setRange("SIG", 0, +10);

   x.setRange("FULL", -10, +10);
   y.setRange("FULL", -10, +10);

   // Try different normalization sets to check if there is a false cache hit
   // after changing the normalization range of the servers.
   RooArgSet normSet1{x, y};
   RooArgSet normSet2{x, y};

   // The fit range needs to be a multi-range to trigger the problem that is covered by this test
   const char *fitRange = "SB1,SB2";

   // Set the normalization range for the top-level model
   model.setNormRange(fitRange);

   const double val1 = model.getVal(normSet1);

   // Now set the normalization range for all other PDFs, which should not
   // change the value of the top-level PDF.
   for (auto *pdf : static_range_cast<RooAbsPdf *>(ws.allPdfs())) {
      pdf->setNormRange(fitRange);
   }

   const double val2 = model.getVal(normSet1);
   const double val3 = model.getVal(normSet2);

   EXPECT_FLOAT_EQ(val2, val1);
   EXPECT_FLOAT_EQ(val3, val1);
}

// Make sure that the pdf that provides the expected number of events in a
// RooProdPdf is not considered as a constraint. Covers JIRA issue ROOT-7604.
TEST(RooProdPdf, RooProdPdfWithExtendedTerm)
{
   RooHelpers::LocalChangeMsgLevel locmsg(RooFit::WARNING, 0u, RooFit::Minimization, false);

   RooWorkspace ws;
   ws.factory("Gaussian::L_constraint(L_nom[10],L[10,0,20],0.2)");
   ws.factory("PROD::full_model( RooExtendedTerm::model( L ) , L_constraint )");

   ws.factory("weightVar[10,0,100]");
   ws.factory("dummy_obs[1,0,1]");
   ws.defineSet("obs", "dummy_obs,weightVar");
   RooDataSet data("data", "data", *ws.set("obs"), RooFit::WeightVar("weightVar"));
   data.add(*ws.set("obs"), 5);

   std::unique_ptr<RooAbsReal> nll1{ws.pdf("full_model")->createNLL(data)};
   double refVal = -std::log(ROOT::Math::gaussian_pdf(10, 0.2, 10)) + (10. - 5 * std::log(10.));
   double nll1Val = nll1->getVal();
   EXPECT_FLOAT_EQ(nll1Val, refVal);
}
