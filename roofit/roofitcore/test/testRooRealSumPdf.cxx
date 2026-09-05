// Tests for RooRealSumPdf
// Authors: Jonas Rembser, CERN 09/2026

#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooPolyVar.h>
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
