// Tests for RooProfileLL
// Authors: Jonas Rembser, CERN 09/2026

#include <RooAddPdf.h>
#include <RooDataSet.h>
#include <RooGaussian.h>
#include <RooMinimizer.h>
#include <RooProfileLL.h>
#include <RooRealVar.h>

#include "gtest_wrapper.h"

#include <memory>
#include <vector>

/// The profile likelihood estimator minimizes the likelihood with respect to
/// all floating parameters except the parameter of interest. This is checked
/// via its defining properties: it vanishes at the best-fit point, it is
/// non-negative, it is bounded from above by the plain likelihood scan, and it
/// agrees with an explicit conditional minimization. All tolerances are at the
/// scale of the Minuit convergence criterion, because RooProfileLL subtracts
/// an absolute minimum from its own internal minimization, which can deviate
/// from the externally found minimum at that level. Replaces the former
/// stressRooFit test based on the rf605 tutorial, which compared likelihood
/// scan curves against stored references.
TEST_P(RooFitEvalBackendTest, ProfileLLMinimization)
{
   using namespace RooFit;

   // Tolerance for comparing results of independent minimizations of the same
   // likelihood, at the scale of the Minuit convergence (EDM) criterion
   const double minimTol = 1e-3;

   RooRealVar x("x", "x", -20, 20);

   // Model with intentionally strong correlations
   RooRealVar mean("mean", "mean of g1 and g2", 0, -10, 10);
   RooGaussian g1("g1", "g1", x, mean, 3.0);
   RooRealVar sigmaG2("sigma_g2", "width of g2", 4, 3.0, 6.0);
   RooGaussian g2("g2", "g2", x, mean, sigmaG2);
   RooRealVar frac("frac", "frac", 0.5, 0.0, 1.0);
   RooAddPdf model("model", "model", {g1, g2}, frac);

   std::unique_ptr<RooDataSet> data{model.generate(x, 1000)};

   std::unique_ptr<RooAbsReal> nll{model.createNLL(*data, _evalBackend)};

   // Find the global minimum
   {
      RooMinimizer m(*nll);
      m.setPrintLevel(-1);
      m.migrad();
   }
   const double minNll = nll->getVal();
   const double fracBest = frac.getVal();

   RooArgSet params{mean, sigmaG2, frac};
   RooArgSet snapshot;
   params.snapshot(snapshot);

   // Plain likelihood scan in frac, with the other parameters fixed to their
   // values at the global minimum. The scan points are chosen far away from
   // the best-fit value of frac (around 0.65), so that the scanned likelihood
   // differences are much larger than the minimization tolerance.
   const std::vector<double> scanPoints{0.2, 0.35, 0.5, 0.8};
   std::vector<double> plainDeltaNll;
   for (double v : scanPoints) {
      frac.setVal(v);
      plainDeltaNll.push_back(nll->getVal() - minNll);
   }
   params.assign(snapshot);

   RooProfileLL pll("pll_frac", "pll_frac", *nll, frac);

   // At the best-fit point the profile likelihood is zero
   frac.setVal(fracBest);
   EXPECT_NEAR(pll.getVal(), 0., minimTol);

   // Away from the minimum, 0 <= profile likelihood <= plain likelihood scan
   for (std::size_t i = 0; i < scanPoints.size(); ++i) {
      frac.setVal(scanPoints[i]);
      const double p = pll.getVal();
      EXPECT_GE(p, -minimTol) << "frac = " << scanPoints[i];
      EXPECT_LE(p, plainDeltaNll[i] + minimTol) << "frac = " << scanPoints[i];
   }

   // Cross-check one profile value against an explicit minimization with the
   // parameter of interest fixed
   const double fracFixed = 0.3;
   frac.setVal(fracFixed);
   const double profileVal = pll.getVal();

   params.assign(snapshot);
   frac.setVal(fracFixed);
   frac.setConstant(true);
   RooMinimizer m2(*nll);
   m2.setPrintLevel(-1);
   m2.migrad();
   EXPECT_NEAR(profileVal, nll->getVal() - minNll, 2. * minimTol);
}

INSTANTIATE_TEST_SUITE_P(RooProfileLL, RooFitEvalBackendTest, testing::Values(ROOFIT_EVAL_BACKENDS),
                         [](testing::TestParamInfo<RooFitEvalBackendTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "EvalBackend" << std::get<0>(paramInfo.param).name();
                            return ss.str();
                         });
