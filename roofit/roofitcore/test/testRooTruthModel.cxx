// Tests for the RooTruthModel
// Authors: Jonas Rembser, CERN 11/2023

#include <RooBDecay.h>
#include <RooConstVar.h>
#include <RooRealVar.h>
#include <RooTruthModel.h>

#include <gtest/gtest.h>

#include <iostream>

/// Check that the integration over a subrange works when using an analytical
/// convolution with the RooTruthModel.
TEST(RooTruthModel, IntegrateSubrange)
{
   using namespace RooFit;

   RooRealVar dt{"dt", "dt", 0, 10};

   RooTruthModel truthModel{"tm", "truth model", dt};

   RooBDecay bcpg{"bcpg0",
                  "bcpg0",
                  dt,
                  RooConst(1.547),
                  RooConst(0.323206),
                  RooConst(1),
                  RooConst(1),
                  RooConst(1.547),
                  RooConst(1.547),
                  RooConst(0.472),
                  truthModel,
                  RooBDecay::DecayType::SingleSided};
   dt.setRange("integral", 2, 2);

   std::unique_ptr<RooAbsReal> integ{bcpg.createIntegral({dt}, "integral")};
   EXPECT_NEAR(integ->getVal(), 0.0, 1e-16);
}
