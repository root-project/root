#include <gtest/gtest.h>

#include "BDTHelpers.hxx"
#include "TMVA/RBDT.hxx"

#include "ROOT/RVec.hxx"

#include <iostream>

using namespace TMVA::Experimental;

TEST(RBDT, ClassificationSingleEvent)
{
   const auto maxDepth = 1;
   const auto numFeatures = 1;
   const auto numTrees = 1;
   WriteModel("myModel", "TestRBDT0.root", "identity", {0}, {0.0, 1.0, -1.0}, {maxDepth}, {numTrees}, {numFeatures},
              {2});

   RBDT bdt("myModel", "TestRBDT0.root");
   auto y = bdt.Compute({-999.0});
   EXPECT_EQ(y.size(), 1u);
   EXPECT_FLOAT_EQ(y[0], 1.0);
   EXPECT_FLOAT_EQ(bdt.Compute({999.0})[0], -1.0);
}

TEST(RBDT, ClassificationSingleEventRVec)
{
   const auto maxDepth = 1;
   const auto numFeatures = 1;
   const auto numTrees = 1;
   WriteModel("myModel", "TestRBDT0.root", "identity", {0}, {0.0, 1.0, -1.0}, {maxDepth}, {numTrees}, {numFeatures},
              {2});

   RBDT bdt("myModel", "TestRBDT0.root");
   ROOT::RVec<float> x = {-999.0};
   auto y = bdt.Compute(x);
   EXPECT_EQ(y.size(), 1u);
   EXPECT_FLOAT_EQ(y[0], 1.0);
}

TEST(RBDT, ClassificationBatch)
{
   const auto maxDepth = 1;
   const auto numFeatures = 1;
   const auto numTrees = 1;
   WriteModel("myModel", "TestRBDT1.root", "identity", {0}, {0.0, 1.0, -1.0}, {maxDepth}, {numTrees}, {numFeatures},
              {2});

   RBDT bdt("myModel", "TestRBDT1.root");
   RTensor<float> x({2, 1});
   x(0, 0) = -999.0;
   x(0, 1) = 999.0;
   auto y = bdt.Compute(x);
   const auto shape = y.GetShape();
   EXPECT_EQ(shape[0], 2u);
   EXPECT_EQ(shape[1], 1u);
   EXPECT_FLOAT_EQ(y(0, 0), 1.0);
   EXPECT_FLOAT_EQ(y(1, 0), -1.0);
}
