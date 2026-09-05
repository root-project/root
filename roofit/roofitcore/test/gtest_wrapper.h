#ifndef RooFit_gtest_wrapper_h
#define RooFit_gtest_wrapper_h

#include <ROOT/RVersion.hxx>

#include <gtest/gtest.h>

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 32, 99)
// Backward compatibility for gtest version < 1.10.0
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif
#endif

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
#define ROOFIT_EVAL_BACKEND_LEGACY RooFit::EvalBackend(RooFit::EvalBackend::Value::Legacy),
#else
#define ROOFIT_EVAL_BACKEND_LEGACY
#endif

#ifdef ROOFIT_CUDA
#define ROOFIT_EVAL_BACKEND_CUDA RooFit::EvalBackend::Cuda(),
#else
#define ROOFIT_EVAL_BACKEND_CUDA
#endif

#ifdef ROOFIT_CLAD
#define ROOFIT_EVAL_BACKEND_CODEGEN RooFit::EvalBackend::Codegen(),
#else
#define ROOFIT_EVAL_BACKEND_CODEGEN
#endif

#define ROOFIT_EVAL_BACKENDS ROOFIT_EVAL_BACKEND_LEGACY ROOFIT_EVAL_BACKEND_CUDA RooFit::EvalBackend::Cpu()

#define ROOFIT_EVAL_BACKENDS_WITH_CODEGEN \
   ROOFIT_EVAL_BACKENDS, ROOFIT_EVAL_BACKEND_CODEGEN RooFit::EvalBackend::CodegenNoGrad()

#include <RooFitResult.h>
#include <RooHelpers.h>
#include <RooRandom.h>
#include <RooRealVar.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cmath>
#include <memory>
#include <tuple>

/// Common fixture for tests that are parametrized over the RooFit evaluation
/// backends: fixes the random seed and silences RooFit messages below WARNING.
class RooFitEvalBackendTest : public testing::TestWithParam<std::tuple<RooFit::EvalBackend>> {
public:
   RooFitEvalBackendTest() : _evalBackend{RooFit::EvalBackend::Legacy()} {}

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

/// Check that the floating fit parameter with the given name is within
/// nSigma fit errors of the truth value.
inline void expectParamNear(RooFitResult const &res, const char *name, double truthVal, double nSigma = 5.)
{
   auto *param = static_cast<RooRealVar const *>(res.floatParsFinal().find(name));
   ASSERT_NE(param, nullptr) << name;
   EXPECT_GT(param->getError(), 0.) << name;
   EXPECT_NEAR(param->getVal(), truthVal, nSigma * param->getError()) << name;
}

MATCHER_P2(RelativeNear, expected, rel_tol,
           "is within relative tolerance around ref=" + ::testing::PrintToString(expected) +
              " (tol=" + ::testing::PrintToString(rel_tol) + ")")
{
   const double diff = std::fabs(arg - expected);
   const double scale = std::fabs(expected);
   // One could also also consider the target value as scale, or an adaptive scale:
   // const double scale = std::max(std::fabs(arg), std::fabs(expected));

   if (diff <= rel_tol * scale)
      return true;

   *result_listener << "error relative to ref = " << diff / scale << ", absolute diff = " << diff;
   return false;
}

#endif // RooFit_gtest_wrapper_h
