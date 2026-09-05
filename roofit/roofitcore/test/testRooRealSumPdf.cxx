// Tests for RooRealSumPdf
// Authors: Jonas Rembser, CERN 09/2026

#include <RooAddPdf.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooPolyVar.h>
#include <RooProdPdf.h>
#include <RooProduct.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>

#include "gtest_wrapper.h"

#include <cmath>
#include <memory>

/// A pdf defined by a sum of real-valued amplitude components. The amplitude
/// values are checked against their analytic expressions, the normalized pdf
/// is checked via normalization-independent value ratios, and a fit to
/// generated data must recover the amplitude strength. Replaces the former
/// stressRooFit test based on the rf704 tutorial, which compared against
/// stored reference plots.
TEST_P(RooFitEvalBackendTest, RealSumPdfAmplitudeSum)
{
   RooRealVar t("t", "time", -1., 15.);
   RooRealVar cosa("cosa", "cos(alpha)", -1., 1.);

   RooRealVar tau("tau", "#tau", 1.5);
   RooRealVar deltaGamma("deltaGamma", "deltaGamma", 0.3);
   RooFormulaVar coshG("coshGBasis", "exp(-@0/ @1)*cosh(@0*@2/2)", {t, tau, deltaGamma});
   RooFormulaVar sinhG("sinhGBasis", "exp(-@0/ @1)*sinh(@0*@2/2)", {t, tau, deltaGamma});

   RooPolyVar poly1("poly1", "poly1", cosa, RooArgList(0.5, 0.2, 0.2), 0);
   RooPolyVar poly2("poly2", "poly2", cosa, RooArgList(1.0, -0.2, 3.0), 0);

   RooProduct ampl1("ampl1", "amplitude 1", {poly1, coshG});
   RooProduct ampl2("ampl2", "amplitude 2", {poly2, sinhG});

   RooRealVar f1("f1", "f1", 1, 0, 2);
   RooRealVar f2("f2", "f2", 0.5, 0, 2);

   RooRealSumPdf pdf("pdf", "pdf", RooArgList(ampl1, ampl2), RooArgList(f1, f2));

   // Analytic expressions for the amplitude components
   auto ampl1Val = [&](double tVal, double cVal) {
      return (0.5 + 0.2 * cVal + 0.2 * cVal * cVal) * std::exp(-tVal / 1.5) * std::cosh(tVal * 0.3 / 2);
   };
   auto ampl2Val = [&](double tVal, double cVal) {
      return (1.0 - 0.2 * cVal + 3.0 * cVal * cVal) * std::exp(-tVal / 1.5) * std::sinh(tVal * 0.3 / 2);
   };

   for (double tVal : {-0.5, 0.3, 2., 8.}) {
      for (double cVal : {-0.9, 0., 0.4}) {
         t.setVal(tVal);
         cosa.setVal(cVal);
         EXPECT_NEAR(ampl1.getVal(), ampl1Val(tVal, cVal), 1e-9) << "(t, cosa) = (" << tVal << ", " << cVal << ")";
         EXPECT_NEAR(ampl2.getVal(), ampl2Val(tVal, cVal), 1e-9) << "(t, cosa) = (" << tVal << ", " << cVal << ")";
      }
   }

   // Ratios of normalized pdf values are independent of the normalization
   // integral and must match the raw amplitude combination
   RooArgSet nset{t, cosa};
   auto normVal = [&](double tVal, double cVal) {
      t.setVal(tVal);
      cosa.setVal(cVal);
      return pdf.getVal(&nset);
   };
   auto rawVal = [&](double tVal, double cVal) {
      return f1.getVal() * ampl1Val(tVal, cVal) + f2.getVal() * ampl2Val(tVal, cVal);
   };
   EXPECT_NEAR(normVal(2., 0.4) / normVal(1., -0.5), rawVal(2., 0.4) / rawVal(1., -0.5), 1e-6);
   EXPECT_NEAR(normVal(6., 0.8) / normVal(2., 0.4), rawVal(6., 0.8) / rawVal(2., 0.4), 1e-6);

   // Generate toy data and fit with one amplitude strength floating. Only the
   // relative amplitude strength is defined, so f1 is kept constant.
   std::unique_ptr<RooDataSet> data{pdf.generate({t, cosa}, 10000)};

   f1.setConstant(true);
   f2.setVal(1.0);

   std::unique_ptr<RooFitResult> res{pdf.fitTo(*data, RooFit::Save(), RooFit::PrintLevel(-1), _evalBackend)};

   EXPECT_EQ(res->status(), 0);
   EXPECT_EQ(res->covQual(), 3);
   expectParamNear(*res, "f2", 0.5);
}

INSTANTIATE_TEST_SUITE_P(RooRealSumPdf, RooFitEvalBackendTest, testing::Values(ROOFIT_EVAL_BACKENDS),
                         [](testing::TestParamInfo<RooFitEvalBackendTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "EvalBackend" << std::get<0>(paramInfo.param).name();
                            return ss.str();
                         });

/// The normalization set that is passed to createExpectedEventsFunc() can
/// contain variables that the pdf does not depend on, for example the channel
/// index when the pdf is a component of a simultaneous mixture. Such variables
/// must not end up in the yield integral: they would factorize out of the
/// integral and scale the expected event count by their range or by their
/// number of states.
TEST(RooRealSumPdf, ExpectedEventsFuncNonDependentsInNormSet)
{
   RooRealVar x("x", "x", 0., 10.);

   // Variables that the pdf doesn't depend on
   RooRealVar y("y", "y", 0., 4.);
   RooCategory channel("channel", "channel", {{"A", 0}, {"B", 1}});

   RooPolyVar shape1("shape1", "shape1", x, RooArgList(0.0, 1.0), 0); // x
   RooPolyVar shape2("shape2", "shape2", x, RooArgList(1.0), 0);      // 1

   RooRealVar yield1("yield1", "yield1", 2.0);
   RooRealVar yield2("yield2", "yield2", 3.0);

   RooRealSumPdf pdf("pdf", "pdf", RooArgList(shape1, shape2), RooArgList(yield1, yield2), true);

   // int_0^10 (2 * x + 3) dx
   const double expected = 2. * 50. + 3. * 10.;

   RooArgSet nset{x};
   RooArgSet nsetExtra{x, y, channel};

   std::unique_ptr<RooAbsReal> func{pdf.createExpectedEventsFunc(&nset)};
   ASSERT_TRUE(func);
   EXPECT_NEAR(func->getVal(), expected, 1e-9);

   std::unique_ptr<RooAbsReal> funcExtra{pdf.createExpectedEventsFunc(&nsetExtra)};
   ASSERT_TRUE(funcExtra);
   EXPECT_FALSE(funcExtra->dependsOn(y));
   EXPECT_FALSE(funcExtra->dependsOn(channel));
   // Without restricting the integration set, y and channel would factorize
   // out of the integral, scaling the yield by 4 * 2.
   EXPECT_NEAR(funcExtra->getVal(), expected, 1e-9);

   // The pdf normalization must not be affected either.
   x.setVal(4.);
   EXPECT_NEAR(pdf.getVal(&nset), (2. * 4. + 3.) / expected, 1e-9);
}

/// Regression test for the crash that motivated restricting the yield
/// integration set. An all-extendable RooAddPdf passes its full normalization
/// set down to the components in compileForNormSet(). When a component only
/// depends on a subset of these observables, the remaining ones used to end up
/// in the component's yield integral, which corrupted the analytical
/// integration cache of the RooRealSumPdf and segfaulted in
/// analyticalIntegralWN() on the first likelihood evaluation.
TEST(RooRealSumPdf, ExpectedEventsFuncInExtendedAddPdf)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x("x", "x", 0., 10.);
   RooRealVar y("y", "y", -5., 5.);

   // Channel A: a RooRealSumPdf in x, times a shape in y
   RooPolyVar a1("a1", "a1", x, RooArgList(0.0, 1.0), 0); // x
   RooPolyVar a2("a2", "a2", x, RooArgList(1.0), 0);      // 1
   RooRealVar yieldA1("yieldA1", "yieldA1", 2.0, 0.01, 100.);
   RooRealVar yieldA2("yieldA2", "yieldA2", 3.0, 0.01, 100.);
   RooRealSumPdf sumA("sumA", "sumA", RooArgList(a1, a2), RooArgList(yieldA1, yieldA2), true);
   RooGenericPdf shapeA("shapeA", "std::exp(-0.5 * (y + 1) * (y + 1))", {y});
   RooProdPdf chanA("chanA", "chanA", RooArgList(sumA, shapeA));

   // Channel B, same structure
   RooPolyVar b1("b1", "b1", x, RooArgList(1.0, -0.05), 0); // 1 - 0.05 * x
   RooRealVar yieldB1("yieldB1", "yieldB1", 4.0, 0.01, 100.);
   RooRealSumPdf sumB("sumB", "sumB", RooArgList(b1), RooArgList(yieldB1), true);
   RooGenericPdf shapeB("shapeB", "std::exp(-0.5 * (y - 2) * (y - 2) / 2.25)", {y});
   RooProdPdf chanB("chanB", "chanB", RooArgList(sumB, shapeB));

   RooAddPdf model("model", "model", RooArgList(chanA, chanB));

   // int_0^10 (2 * x + 3) dx + int_0^10 4 * (1 - 0.05 * x) dx
   const double nExpected = 130. + 30.;

   RooArgSet nset{x, y};
   EXPECT_NEAR(model.expectedEvents(&nset), nExpected, 1e-9);

   RooDataSet data("data", "data", nset);
   for (int i = 0; i < 5; ++i) {
      x.setVal(1.0 + 1.5 * i);
      y.setVal(-2.0 + i);
      data.add(nset);
   }
   const double nObserved = data.numEntries();

   std::unique_ptr<RooAbsReal> nll{model.createNLL(data, RooFit::Extended(true), RooFit::EvalBackend::Cpu())};
   const double nllVal = nll->getVal();
   ASSERT_TRUE(std::isfinite(nllVal));

   // Scaling all yields by a common factor leaves the normalized shapes
   // untouched, so the extended likelihood must shift by exactly
   // (k - 1) * nExpected - nObserved * log(k). This pins down the expected
   // event count that the compiled likelihood actually used.
   const double k = 2.0;
   yieldA1.setVal(k * yieldA1.getVal());
   yieldA2.setVal(k * yieldA2.getVal());
   yieldB1.setVal(k * yieldB1.getVal());

   EXPECT_NEAR(nll->getVal() - nllVal, (k - 1.) * nExpected - nObserved * std::log(k), 1e-6);
}
