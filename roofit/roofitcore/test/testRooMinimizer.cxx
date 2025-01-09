// Tests for the RooProdPdf
// Author: Jonas Rembser, CERN, October 2024

#include <RooAddPdf.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooRandom.h>
#include <RooRealVar.h>

#include "gtest_wrapper.h"

class EvalBackendParametrizedTest : public testing::TestWithParam<std::tuple<RooFit::EvalBackend>> {
public:
   EvalBackendParametrizedTest() : _evalBackend{RooFit::EvalBackend::Legacy()} {}

private:
   void SetUp() override
   {
      RooRandom::randomGenerator()->SetSeed(1337ul);
      _evalBackend = std::get<0>(GetParam());
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING);
   }

   void TearDown() override { _changeMsgLvl.reset(); }

protected:
   RooFit::EvalBackend _evalBackend;

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

// Unit test corresponding to the rf601 tutorial, but parametrized for the
// different evaluation backends.
TEST_P(EvalBackendParametrizedTest, RF601)
{
   RooRealVar x("x", "x", -20, 20);

   // Model (intentional strong correlations)
   RooRealVar mean("mean", "mean of g1 and g2", 0);
   RooRealVar sigma_g1("sigma_g1", "width of g1", 3, 1.0, 5.0);
   RooGaussian g1("g1", "g1", x, mean, sigma_g1);

   RooRealVar sigma_g2("sigma_g2", "width of g2", 4, 3.0, 6.0);
   RooGaussian g2("g2", "g2", x, mean, sigma_g2);

   RooRealVar frac("frac", "frac", 0.5, 0.0, 1.0);
   RooAddPdf model("model", "model", RooArgList(g1, g2), frac);

   std::unique_ptr<RooDataSet> data{model.generate(x, 1000)};

   std::unique_ptr<RooAbsReal> nll{model.createNLL(*data, RooFit::EvalBackend(_evalBackend))};

   // Reference fit results. We are building them manually in this code in
   // order to avoid binary reference files.
   RooFitResult rRef;
   RooFitResult r2Ref;

   rRef.setInitParList({frac, sigma_g1, sigma_g2});
   r2Ref.setInitParList({frac, sigma_g1});

   {
      RooArgList params{frac, sigma_g1, sigma_g2};
      rRef.setMinNLL(2690.1655479975625);
      std::vector<double> valsRef{0.48732473752392391, 2.7579472102402196, 4.2132158568489011};
      std::vector<double> errorsRef{0.29415767363636769, 0.48854402667226893, 0.47322653666757764};
      std::vector<double> globalCC{0.98208607303266726, 0.94654805382506446, 0.9542122909792754};
      std::vector<double> corrsV{1.0, 0.927913, 0.938339, 0.927913, 1.0, 0.806083, 0.938339, 0.806083, 1.0};
      const std::size_t nParams = params.size();

      RooArgList paramsPostFit;
      params.snapshot(paramsPostFit, false);

      for (std::size_t i = 0; i < nParams; ++i) {
         auto &var = static_cast<RooRealVar &>(paramsPostFit[i]);
         var.setVal(valsRef[i]);
         var.setError(errorsRef[i]);
      }

      rRef.setConstParList(mean);
      rRef.setFinalParList(paramsPostFit);
      TMatrixDSym corrs(nParams);
      TMatrixDSym covs(nParams);
      corrs.SetMatrixArray(corrsV.data());
      rRef.fillCorrMatrix(globalCC, corrs, covs);
   }

   RooMinimizer m(*nll);

   m.setPrintLevel(-1);

   m.migrad();
   m.hesse();

   m.minos(sigma_g2); // Run MINOS on sigma_g2 parameter only

   std::unique_ptr<RooFitResult> r{m.save()};

   // You can manually change the value of a (constant) parameter
   mean = 0.3;

   {
      RooArgList params{frac, sigma_g1};
      r2Ref.setMinNLL(2698.6818728208696);
      std::vector<double> constValsRef{0.3, 4.1869317966930382};
      std::vector<double> constErrorsRef{0.0, 0.0};
      std::vector<double> valsRef{0.45212322018672196, 2.746677666046311};
      std::vector<double> errorsRef{0.11175880034776256, 0.31839226266447529};
      std::vector<double> globalCC{0.84401375662480127, 0.84401375662480116, 0.};
      std::vector<double> corrsV{1.0, 0.84401326379874853, 0, 0.84401326379874853, 1., 0., 0., 0.};

      RooArgList constParams;
      RooArgList{mean, sigma_g2}.snapshot(constParams, false);

      for (std::size_t i = 0; i < constParams.size(); ++i) {
         auto &var = static_cast<RooRealVar &>(constParams[i]);
         var.setVal(constValsRef[i]);
         var.setError(constErrorsRef[i]);
      }

      RooArgList paramsPostFit;
      params.snapshot(paramsPostFit, false);

      for (std::size_t i = 0; i < paramsPostFit.size(); ++i) {
         auto &var = static_cast<RooRealVar &>(paramsPostFit[i]);
         var.setVal(valsRef[i]);
         var.setError(errorsRef[i]);
      }

      r2Ref.setConstParList(constParams);
      r2Ref.setFinalParList(paramsPostFit);
      TMatrixDSym corrs(3);
      TMatrixDSym covs(3);
      corrs.SetMatrixArray(corrsV.data());
      r2Ref.fillCorrMatrix(globalCC, corrs, covs);
   }

   // Rerun MIGRAD,HESSE
   m.migrad();
   m.hesse();

   // Now fix sigma_g2
   sigma_g2.setConstant(true);

   // Rerun MIGRAD,HESSE
   m.migrad();
   m.hesse();

   std::unique_ptr<RooFitResult> r2{m.save()};

   // The tolerance parameter is necessary because not all backends give
   // exactly the same results: when using AD, the final result is slightly
   // different.
   const double tol = 1e-4;

   EXPECT_TRUE(r->isIdentical(rRef, tol, tol));
   EXPECT_TRUE(r2->isIdentical(r2Ref, tol, tol));
}

INSTANTIATE_TEST_SUITE_P(RooMinimizer, EvalBackendParametrizedTest, testing::Values(ROOFIT_EVAL_BACKENDS_WITH_CODEGEN),
                         [](testing::TestParamInfo<EvalBackendParametrizedTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "EvalBackend" << std::get<0>(paramInfo.param).name();
                            return ss.str();
                         });
