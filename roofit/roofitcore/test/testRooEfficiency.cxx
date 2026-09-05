// Tests for RooEfficiency and RooEffProd
// Authors: Jonas Rembser, CERN 09/2026

#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooEffProd.h>
#include <RooEfficiency.h>
#include <RooExponential.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooPolynomial.h>
#include <RooProdPdf.h>
#include <RooRealVar.h>

#include "gtest_wrapper.h"

#include <cmath>
#include <memory>

/// The value of the efficiency pdf normalized over the acceptance category
/// must reproduce the efficiency function for the "accept" state and its
/// complement for the "reject" state.
TEST(RooEfficiency, ValueVsEfficiencyFunction)
{
   RooRealVar x("x", "x", -10, 10);
   RooRealVar a("a", "a", 0.4, 0, 1);
   RooRealVar b("b", "b", 5);
   RooRealVar c("c", "c", -1, -10, 10);
   RooFormulaVar effFunc("effFunc", "(1-a)+a*cos((x-c)/b)", RooArgList(a, b, c, x));

   RooCategory cut("cut", "cut", {{"accept", 1}, {"reject", 0}});
   RooEfficiency effPdf("effPdf", "effPdf", effFunc, cut, "accept");

   RooArgSet nset{cut};
   for (double xVal : {-9.5, -5., -1., 0., 2.5, 7., 9.5}) {
      x.setVal(xVal);
      cut.setLabel("accept");
      EXPECT_NEAR(effPdf.getVal(&nset), effFunc.getVal(), 1e-9) << "x = " << xVal;
      cut.setLabel("reject");
      EXPECT_NEAR(effPdf.getVal(&nset), 1. - effFunc.getVal(), 1e-9) << "x = " << xVal;
   }
}

/// Unbinned maximum likelihood fit of an efficiency function to a dataset
/// D(x, cut), with the shape observable as conditional observable. The fit
/// must recover the efficiency parameters used for generation. Replaces the
/// former stressRooFit test based on the rf701 tutorial.
TEST_P(RooFitEvalBackendTest, ConditionalFit1D)
{
   using namespace RooFit;

   RooRealVar x("x", "x", -10, 10);

   RooRealVar a("a", "a", 0.4, 0, 1);
   RooRealVar b("b", "b", 5);
   RooRealVar c("c", "c", -1, -10, 10);
   RooFormulaVar effFunc("effFunc", "(1-a)+a*cos((x-c)/b)", RooArgList(a, b, c, x));

   RooCategory cut("cut", "cut", {{"accept", 1}, {"reject", 0}});
   RooEfficiency effPdf("effPdf", "effPdf", effFunc, cut, "accept");

   // Toy model eff(cut|x) * shape(x), only needed for generation
   RooPolynomial shapePdf("shapePdf", "shapePdf", x, RooConst(-0.095));
   RooProdPdf model("model", "model", shapePdf, Conditional(effPdf, cut));

   std::unique_ptr<RooDataSet> data{model.generate({x, cut}, 10000)};

   std::unique_ptr<RooFitResult> res{
      effPdf.fitTo(*data, ConditionalObservables(x), Save(), PrintLevel(-1), _evalBackend)};

   EXPECT_EQ(res->status(), 0);
   expectParamNear(*res, "a", 0.4);
   expectParamNear(*res, "c", -1.);
}

/// Two-dimensional version of the conditional efficiency fit. Replaces the
/// former stressRooFit test based on the rf702 tutorial.
TEST_P(RooFitEvalBackendTest, ConditionalFit2D)
{
   using namespace RooFit;

   RooRealVar x("x", "x", -10, 10);
   RooRealVar y("y", "y", -10, 10);

   RooRealVar ax("ax", "ax", 0.6, 0, 1);
   RooRealVar bx("bx", "bx", 5);
   RooRealVar cx("cx", "cx", -1, -10, 10);
   RooRealVar ay("ay", "ay", 0.2, 0, 1);
   RooRealVar by("by", "by", 5);
   RooRealVar cy("cy", "cy", -1, -10, 10);

   RooFormulaVar effFunc("effFunc", "((1-ax)+ax*cos((x-cx)/bx))*((1-ay)+ay*cos((y-cy)/by))",
                         RooArgList(ax, bx, cx, x, ay, by, cy, y));

   RooCategory cut("cut", "cut", {{"accept", 1}, {"reject", 0}});
   RooEfficiency effPdf("effPdf", "effPdf", effFunc, cut, "accept");

   RooPolynomial shapePdfX("shapePdfX", "shapePdfX", x, RooConst(-0.095));
   RooPolynomial shapePdfY("shapePdfY", "shapePdfY", y, RooConst(+0.095));
   RooProdPdf shapePdf("shapePdf", "shapePdf", RooArgSet(shapePdfX, shapePdfY));
   RooProdPdf model("model", "model", shapePdf, Conditional(effPdf, cut));

   std::unique_ptr<RooDataSet> data{model.generate({x, y, cut}, 10000)};

   std::unique_ptr<RooFitResult> res{
      effPdf.fitTo(*data, ConditionalObservables(RooArgSet(x, y)), Save(), PrintLevel(-1), _evalBackend)};

   EXPECT_EQ(res->status(), 0);
   expectParamNear(*res, "ax", 0.6);
   expectParamNear(*res, "cx", -1.);
   expectParamNear(*res, "ay", 0.2);
   expectParamNear(*res, "cy", -1.);
}

/// A pdf multiplied with an acceptance efficiency via RooEffProd. The
/// normalized value is checked against the analytically known shape, and a fit
/// to generated data must recover the decay constant. Replaces the former
/// stressRooFit test based on the rf703 tutorial.
TEST_P(RooFitEvalBackendTest, EffProdFit)
{
   using namespace RooFit;

   if (_evalBackend.value() == EvalBackend::Value::Codegen || _evalBackend.value() == EvalBackend::Value::CodegenNoGrad) {
      GTEST_SKIP() << "RooEffProd is not supported by the codegen backend yet";
   }

   RooRealVar t("t", "t", 0, 5);
   RooRealVar tau("tau", "tau", -1.54, -4, -0.1);
   RooExponential model("model", "model", t, tau);

   // Error function to simulate a turn-on slope
   RooFormulaVar eff("eff", "0.5*(std::erf((t-1)/0.5)+1)", t);
   RooEffProd modelEff("modelEff", "model with efficiency", model, eff);

   // The ratio of normalized pdf values is independent of the normalization
   // integral and must match exp(tau * t) * eff(t) ratios
   auto rawVal = [&](double tVal) { return std::exp(tau.getVal() * tVal) * 0.5 * (std::erf((tVal - 1.) / 0.5) + 1.); };
   RooArgSet nset{t};
   auto normVal = [&](double tVal) {
      t.setVal(tVal);
      return modelEff.getVal(&nset);
   };
   EXPECT_NEAR(normVal(1.5) / normVal(0.8), rawVal(1.5) / rawVal(0.8), 1e-6);
   EXPECT_NEAR(normVal(3.0) / normVal(1.5), rawVal(3.0) / rawVal(1.5), 1e-6);

   std::unique_ptr<RooDataSet> data{modelEff.generate(t, 10000)};

   std::unique_ptr<RooFitResult> res{modelEff.fitTo(*data, Save(), PrintLevel(-1), _evalBackend)};

   EXPECT_EQ(res->status(), 0);
   expectParamNear(*res, "tau", -1.54);
}

INSTANTIATE_TEST_SUITE_P(RooEfficiency, RooFitEvalBackendTest, testing::Values(ROOFIT_EVAL_BACKENDS_WITH_CODEGEN),
                         [](testing::TestParamInfo<RooFitEvalBackendTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "EvalBackend" << std::get<0>(paramInfo.param).name();
                            return ss.str();
                         });
