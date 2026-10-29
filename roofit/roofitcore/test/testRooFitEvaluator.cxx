// Tests for RooFit::Evaluator
// Authors: Jonas Rembser, CERN 10/2026

#include <RooFit/Evaluator.h>
#include <RooAddition.h>
#include <RooRealVar.h>

#include "gtest_wrapper.h"

#include <memory>

/// The Evaluator is a client of all nodes in its computation graph, to know in
/// its destructor which nodes are still alive. Check that it registers and
/// unregisters itself cleanly, and that a node that dies before the Evaluator
/// is not touched anymore. The latter is a use-after-free that only sanitizers
/// or valgrind can see, but it's what crashed Python sessions at exit when the
/// garbage collector destroyed a parameter before the likelihood.
TEST(RooFitEvaluator, NodeDiesBeforeEvaluator)
{
   RooRealVar x{"x", "x", 0.0, -10.0, 10.0};
   auto mean = std::make_unique<RooRealVar>("mean", "mean", 0.0, -10.0, 10.0);
   RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 10.0};
   RooAddition sum{"sum", "sum", {x, *mean, sigma}};

   {
      RooFit::Evaluator evaluator{sum};
      EXPECT_DOUBLE_EQ(evaluator.run()[0], sum.getVal());

      // The Evaluator is a client of the nodes, but not a value or shape
      // client, so it doesn't take part in the dirty state propagation.
      EXPECT_EQ(x.clients().size(), 2);
      EXPECT_EQ(x.valueClients().size(), 1);
      EXPECT_EQ(x.shapeClients().size(), 0);

      // The parameter dies before the Evaluator.
      mean.reset();
   }

   // The Evaluator unregistered itself from the surviving nodes.
   EXPECT_EQ(x.clients().size(), 1);
   EXPECT_EQ(sigma.clients().size(), 1);
   EXPECT_EQ(sum.clients().size(), 0);
}
