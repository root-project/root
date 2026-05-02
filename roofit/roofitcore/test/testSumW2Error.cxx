// Tests for the SumW2Error correction
// Author: Jonas Rembser, CERN  10/2021

#include <RooFitResult.h>
#include <RooAbsPdf.h>
#include <RooAddPdf.h>
#include <RooRandom.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooHelpers.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <string>

namespace {

// Compare two fit results parameter-by-parameter with relative tolerances on
// values and errors. The error of each parameter in `ref` is multiplied by
// `errorScale[name]` (default 1.0) before being compared to the same parameter
// in `r`.
void expectFitsCompatible(const RooFitResult &ref, const RooFitResult &r, double valTol, double errTol,
                          const std::map<std::string, double> &errorScale = {})
{
   ASSERT_EQ(ref.floatParsFinal().size(), r.floatParsFinal().size());

   for (auto *p : ref.floatParsFinal()) {
      const std::string name = p->GetName();
      auto *vRef = static_cast<RooRealVar *>(p);
      auto *vNew = static_cast<RooRealVar *>(r.floatParsFinal().find(name.c_str()));
      ASSERT_NE(vNew, nullptr) << "missing parameter " << name;

      const double valScale = std::max(std::abs(vRef->getVal()), 1.0);
      EXPECT_NEAR(vRef->getVal(), vNew->getVal(), valTol * valScale) << "value mismatch for " << name;

      const double scale = errorScale.count(name) ? errorScale.at(name) : 1.0;
      EXPECT_NEAR(vRef->getError() * scale, vNew->getError(), errTol * vRef->getError() * scale)
         << "error mismatch for " << name;
   }
}

} // namespace

// GitHub issue 9118: Problem running weighted binned fit in batch mode.
//
// Test the SumW2Error correction for a non-extended fit by checking the
// defining property of the correction: applied to data with a uniform weight
// w, it should reproduce the parameter values and errors of the equivalent
// unweighted fit. In a non-extended fit, all parameters are shape parameters
// that are invariant under uniform reweighting, so:
//   - Values are unchanged whether the data is weighted or not.
//   - Errors with SumW2(true) match the unweighted-fit errors.
//   - Errors with SumW2(false) on weighted data scale as 1/sqrt(w), because
//     the (unrescaled) Hessian of the likelihood scales linearly with w.
TEST(SumW2Error, BatchMode)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws{"workspace"};
   ws.factory("Gaussian::sig(x[0,0,10],mu[3,0,10],s[1, 0.1, 5])");
   ws.factory("Exponential::bkg(x,c1[-0.5, -3, -0.1])");
   ws.factory("SUM::model(f[0.2, 0.0, 1.0] * sig, bkg)");

   auto &model = *ws.pdf("model");

   RooRandom::randomGenerator()->SetSeed(4357);
   std::unique_ptr<RooDataSet> dataSet{model.generate(*ws.var("x"), 1000)};

   RooArgSet params;
   RooArgSet initialParams;

   model.getParameters(dataSet->get(), params);
   params.snapshot(initialParams);

   const double w = 0.5;

   RooDataSet dataSetWeighted("dataSetWeighted", "dataSetWeighted", *dataSet->get(), RooFit::WeightVar());
   for (int i = 0; i < dataSet->numEntries(); ++i) {
      dataSetWeighted.add(*dataSet->get(i), w);
   }

   std::unique_ptr<RooDataHist> dataHist{dataSet->binnedClone()};
   std::unique_ptr<RooDataHist> dataHistWeighted{dataSetWeighted.binnedClone()};

   using namespace RooFit;

   auto fit = [&](RooAbsData &data, bool sumw2) {
      params.assign(initialParams);
      return std::unique_ptr<RooFitResult>{
         model.fitTo(data, Save(), SumW2Error(sumw2), Strategy(1), EvalBackend::Cpu(), PrintLevel(-1))};
   };

   const double valTol = 1e-4;
   const double errTol = 1e-2;

   for (const auto &dataPair : std::vector<std::pair<RooAbsData *, RooAbsData *>>{
           {dataSet.get(), &dataSetWeighted}, {dataHist.get(), dataHistWeighted.get()}}) {
      RooAbsData &dataUw = *dataPair.first;
      RooAbsData &dataW = *dataPair.second;

      // Reference: fit the unweighted dataset without SumW2Error.
      auto refFit = fit(dataUw, /*sumw2=*/false);

      // SumW2Error on unweighted data should be a near no-op (the internal
      // sqrt(w)-weighted refit is the same fit), so values and errors
      // should match the reference.
      auto fitUwSw2 = fit(dataUw, /*sumw2=*/true);
      expectFitsCompatible(*refFit, *fitUwSw2, valTol, errTol);

      // SumW2Error on weighted data should reproduce the unweighted errors.
      auto fitWSw2 = fit(dataW, /*sumw2=*/true);
      expectFitsCompatible(*refFit, *fitWSw2, valTol, errTol);

      // Without SumW2Error, the errors on a uniformly weighted dataset
      // scale as 1/sqrt(w) for shape parameters.
      auto fitWNoSw2 = fit(dataW, /*sumw2=*/false);
      std::map<std::string, double> scale;
      for (auto *p : refFit->floatParsFinal()) {
         scale[p->GetName()] = 1.0 / std::sqrt(w);
      }
      expectFitsCompatible(*refFit, *fitWNoSw2, valTol, errTol, scale);
   }
}

// Test the SumW2Error correction in an extended fit, in both the full range
// and a subrange. The defining property of SumW2Error is the same as in the
// non-extended case (it should reproduce the unweighted-fit errors), but for
// extended fits the yield parameters scale linearly with the uniform weight
// w under reweighting, and so do their errors.
TEST(SumW2Error, ExtendedFit)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   RooWorkspace ws("workspace");
   ws.factory("x[-10, 10]");
   ws.factory("Gaussian::sig(x, mu[-1, 1], s[0.1, 5])");
   ws.factory("Chebychev::bkg(x, {c1[0.1, -1, 1]})");
   ws.factory("SUM::shp(Nsig[0, 20000] * sig, Nbkg[0, 20000] * bkg)");
   auto *x = ws.var("x");
   x->setRange("subrange", -5.0, 5.0);
   auto *shp = ws.pdf("shp");
   std::unique_ptr<RooDataSet> dataNoWeights{shp->generate(RooArgSet(*x))};

   const double w = 1.5;
   auto *wFunc = ws.factory("w[1.5]");
   auto *wcol = dataNoWeights->addColumn(*wFunc);
   RooDataSet dataWeighted{dataNoWeights->GetName(), dataNoWeights->GetTitle(), *dataNoWeights->get(),
                           RooFit::Import(*dataNoWeights), RooFit::WeightVar(wcol->GetName())};
   RooDataHist datahistUw{"datahistUw", "", *dataNoWeights->get(), *dataNoWeights};
   RooDataHist datahistW{"datahistW", "", *dataWeighted.get(), dataWeighted};

   RooArgSet params;
   RooArgSet initialParams;
   shp->getParameters(dataNoWeights->get(), params);
   params.snapshot(initialParams);

   auto doFit = [&](RooAbsData &data, bool sumw2, const char *range) {
      params.assign(initialParams);
      return std::unique_ptr<RooFitResult>{shp->fitTo(data, Extended(), Range(range), Save(), SumW2Error(sumw2),
                                                      Strategy(1), PrintLevel(-1), EvalBackend::Cpu())};
   };

   // Build a "yield-scaled" reference where yield values and errors are
   // multiplied by `factor`. Shape parameters are left unchanged. This
   // produces the expected fit result for a uniformly weighted dataset under
   // the SumW2 correction.
   auto scaleYields = [&](RooFitResult const &r, double factor) {
      auto out = std::unique_ptr<RooFitResult>(static_cast<RooFitResult *>(r.Clone()));
      for (auto *p : out->floatParsFinal()) {
         const std::string name = p->GetName();
         if (name == "Nsig" || name == "Nbkg") {
            auto *v = static_cast<RooRealVar *>(p);
            v->setVal(v->getVal() * factor);
            v->setError(v->getError() * factor);
         }
      }
      return out;
   };

   for (const char *range : {static_cast<const char *>(nullptr), "subrange"}) {
      // The unweighted reference fit. Yield parameters in the model are
      // defined over the full range of x (the model integrates over the full
      // range internally), so for both `nullptr` and `"subrange"` the yield
      // values represent total counts in [-10, 10].
      auto refFit = doFit(datahistUw, /*sumw2=*/false, range);

      // Tolerances are looser in the subrange because the fit is harder
      // (smaller effective sample size and stronger correlations).
      const bool isSubrange = range != nullptr;
      const double valTol = isSubrange ? 5e-3 : 1e-3;
      const double errTol = isSubrange ? 5e-2 : 2e-2;

      // The reference scaled by w in the yield values and errors. With
      // SumW2(true) on uniformly weighted data, this matches the actual fit:
      //   - yield values scale by w (more weighted "counts" by factor w)
      //   - yield errors also scale by w (fluctuations of weighted counts)
      //   - shape parameters are invariant under uniform reweighting
      auto refScaledByW = scaleYields(*refFit, w);

      auto fitWSw2 = doFit(datahistW, /*sumw2=*/true, range);
      expectFitsCompatible(*refScaledByW, *fitWSw2, valTol, errTol);

      // Without SumW2, the yield values still scale by w, but the errors of
      // *all* parameters get an additional factor 1/sqrt(w) relative to the
      // yield-scaled reference. For yields this means the actual error scales
      // as sqrt(w) (instead of w). For shape parameters this means the error
      // scales as 1/sqrt(w) (instead of 1). This follows from the fact that
      // the weighted-likelihood Hessian scales linearly with the weight.
      std::map<std::string, double> scaleNoSw2;
      for (auto *p : refFit->floatParsFinal()) {
         scaleNoSw2[p->GetName()] = 1.0 / std::sqrt(w);
      }

      auto fitWNoSw2 = doFit(datahistW, /*sumw2=*/false, range);
      expectFitsCompatible(*refScaledByW, *fitWNoSw2, valTol, errTol, scaleNoSw2);
   }
}
