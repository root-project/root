// Tests for RooFit::Evaluator
// Author: Jonas Rembser, CERN 10/2026

#include <RooAddition.h>
#include <RooConstraintSum.h>
#include <RooFit/Evaluator.h>
#include <RooGaussian.h>
#include <RooRealVar.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

/// Deleting nodes of the computation graph before the Evaluator must not
/// crash: the Evaluator used to reset the data tokens of its nodes in the
/// destructor without checking whether the nodes still exist (use after free
/// that was observed as a `free(): invalid pointer` abort at the exit of
/// Python scripts, where the garbage collector can delete the RooWorkspace
/// owning the model before the NLL object).
TEST(RooEvaluator, NodesDeletedBeforeEvaluator)
{
   auto x = std::make_unique<RooRealVar>("x", "x", 0.0, -5.0, 5.0);
   auto mu = std::make_unique<RooRealVar>("mu", "mu", 0.0, -1.0, 1.0);
   auto sigma = std::make_unique<RooRealVar>("sigma", "sigma", 1.0, 0.1, 10.0);
   auto gauss = std::make_unique<RooGaussian>("gauss", "gauss", *x, *mu, *sigma);

   auto evaluator = std::make_unique<RooFit::Evaluator>(*gauss);
   EXPECT_FLOAT_EQ(evaluator->run()[0], gauss->getVal());

   // Delete the pdf before the Evaluator that holds a data token for it.
   gauss.reset();
   evaluator.reset();

   // A new model with the same node name must get a fresh data token.
   auto gauss2 = std::make_unique<RooGaussian>("gauss", "gauss", *x, *mu, *sigma);
   {
      RooFit::Evaluator evaluator2{*gauss2};
      EXPECT_FLOAT_EQ(evaluator2.run()[0], gauss2->getVal());
   }

   // The other order: the node outlives the Evaluator. Its data token must
   // have been reset such that it can be used by a new Evaluator.
   RooFit::Evaluator evaluator3{*gauss2};
   EXPECT_FLOAT_EQ(evaluator3.run()[0], gauss2->getVal());
}

/// A node outputs its logarithm only if its single client can use it, and if
/// its values are not given as input.
TEST(RooEvaluator, LogEvaluation)
{
   RooRealVar x{"x", "x", 40.0, -50.0, 50.0};
   RooRealVar mu{"mu", "mu", 0.0, -1.0, 1.0};
   RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 10.0};
   RooGaussian gauss{"gauss", "gauss", x, mu, sigma};
   RooConstraintSum constraint{"constraint", "constraint", gauss, {}};

   // The Gaussian underflows, but not its logarithm.
   EXPECT_EQ(gauss.getVal(), 0.0);
   EXPECT_DOUBLE_EQ(RooFit::Evaluator{constraint}.run()[0], 800.0);

   // The sum is a second client, which needs the value of the Gaussian.
   x.setVal(1.0);
   RooAddition sum{"sum", "sum", {constraint, gauss}};
   EXPECT_DOUBLE_EQ(RooFit::Evaluator{sum}.run()[0], 0.5 + std::exp(-0.5));

   RooFit::Evaluator evaluator{constraint};
   std::vector<double> input{0.25};
   evaluator.setInput("gauss", input, false);
   EXPECT_DOUBLE_EQ(evaluator.run()[0], -std::log(0.25));
}
