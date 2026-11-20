// Test cuda mode without visible GPUs.
// Author: Stephan Hageboeck, CERN 11/2026

#include <RooDataSet.h>
#include <RooFit/Evaluator.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooRealVar.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>

// RooFit::Evaluator with the cuda toolkit visible but without GPUs used to throw
// during a constructor, which didn't correctly deregister the partly created nodes
// from the computation graph. This led to a crash.
TEST(RooEvaluatorNoGPU, CreateNLLThrowsCleanly)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x("x", "x", 0.0, -5.0, 5.0);
   RooRealVar mu("mu", "mu", 0.0, -1.0, 1.0);
   RooRealVar sigma("sigma", "sigma", 1.0, 0.1, 10.0);
   RooGaussian gauss("gauss", "gauss", x, mu, sigma);

   {
      try {
         RooFit::Evaluator evaluator{gauss, /*useGPU=*/true};
         GTEST_SKIP() << "A GPU is visible, so the Evaluator in CUDA mode won't throw.";
      } catch (std::runtime_error const &) {
         // This should not crash
      }
   }

   std::unique_ptr<RooDataSet> data{gauss.generate(x, 100)};

   EXPECT_THROW(std::unique_ptr<RooAbsReal>(gauss.createNLL(*data, RooFit::EvalBackend::Cuda())), std::runtime_error);

   std::unique_ptr<RooAbsReal> nllCpu{gauss.createNLL(*data, RooFit::EvalBackend::Cpu())};
   EXPECT_TRUE(std::isfinite(nllCpu->getVal()));
}
