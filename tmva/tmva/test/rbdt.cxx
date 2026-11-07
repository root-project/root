#include <gtest/gtest.h>

#include "BDTHelpers.hxx"
#include "TMVA/RBDT.hxx"

#include "ROOT/RVec.hxx"

#include <cmath>
#include <vector>

using namespace TMVA::Experimental;

TEST(RBDT, ClassificationSingleEvent)
{
   const auto maxDepth = 1;
   const auto numInputs = 1;
   const auto numTrees = 1;
   WriteModel("myModel", "TestRBDT0.root", "identity", {0}, {0}, {0.0, 1.0, -1.0}, {maxDepth}, {numTrees}, {numInputs},
              {1});

   RBDT<> bdt("myModel", "TestRBDT0.root");
   auto y = bdt.Compute({-999.0});
   EXPECT_EQ(y.size(), 1u);
   EXPECT_FLOAT_EQ(y[0], 1.0);
   EXPECT_FLOAT_EQ(bdt.Compute({999.0})[0], -1.0);
}

TEST(RBDT, ClassificationSingleEventRVec)
{
   const auto maxDepth = 1;
   const auto numInputs = 1;
   const auto numTrees = 1;
   WriteModel("myModel", "TestRBDT1.root", "identity", {0}, {0}, {0.0, 1.0, -1.0}, {maxDepth}, {numTrees}, {numInputs},
              {1});

   RBDT<> bdt("myModel", "TestRBDT1.root");
   ROOT::RVec<float> x = {-999.0};
   auto y = bdt.Compute(x);
   EXPECT_EQ(y.size(), 1u);
   EXPECT_FLOAT_EQ(y[0], 1.0);
}

TEST(RBDT, ClassificationBatch)
{
   const auto maxDepth = 1;
   const auto numInputs = 1;
   const auto numTrees = 1;
   WriteModel("myModel", "TestRBDT2.root", "identity", {0}, {0}, {0.0, 1.0, -1.0}, {maxDepth}, {numTrees}, {numInputs},
              {1});

   RBDT<> bdt("myModel", "TestRBDT2.root");
   const std::vector<float> x = {-999.0, 999.0};
   auto y = bdt.Compute(x, 1);
   EXPECT_EQ(y.size(), 2u);
   EXPECT_FLOAT_EQ(y[0], 1.0);
   EXPECT_FLOAT_EQ(y[1], -1.0);
}

TEST(RBDT, MulticlassSingleEvent)
{
   const auto maxDepth = 1;
   const auto numInputs = 1;
   const auto numOutputs = 3;
   const auto numTrees = 3;
   WriteModel("myModel", "TestRBDT3.root", "softmax", {0, 0, 0}, {0, 1, 2},
              {0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 2.0, -2.0}, {maxDepth}, {numTrees}, {numInputs}, {numOutputs});

   RBDT<> bdt("myModel", "TestRBDT3.root");

   auto y = bdt.Compute({-999.0});
   EXPECT_EQ(y.size(), 3u);
   const auto s = std::exp(1.0) + std::exp(-1.0) + std::exp(2.0);
   EXPECT_FLOAT_EQ(y[0], std::exp(1.0) / s);
   EXPECT_FLOAT_EQ(y[1], std::exp(-1.0) / s);
   EXPECT_FLOAT_EQ(y[2], std::exp(2.0) / s);

   auto y2 = bdt.Compute({999.0});
   EXPECT_EQ(y2.size(), 3u);
   const auto s2 = std::exp(-1.0) + std::exp(1.0) + std::exp(-2.0);
   EXPECT_FLOAT_EQ(y2[0], std::exp(-1.0) / s2);
   EXPECT_FLOAT_EQ(y2[1], std::exp(1.0) / s2);
   EXPECT_FLOAT_EQ(y2[2], std::exp(-2.0) / s2);
}

TEST(RBDT, MulticlassBatch)
{
   const auto maxDepth = 1;
   const auto numInputs = 1;
   const auto numOutputs = 3;
   const auto numTrees = 3;
   WriteModel("myModel", "TestRBDT4.root", "identity", {0, 0, 0}, {0, 1, 2},
              {0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 2.0, -2.0}, {maxDepth}, {numTrees}, {numInputs}, {numOutputs});

   RBDT<> bdt("myModel", "TestRBDT4.root");
   const std::vector<float> x = {-999.0, 999.0};
   auto y = bdt.Compute(x, 1);
   EXPECT_EQ(y.size(), 6u);
   EXPECT_FLOAT_EQ(y[0], 1.0);
   EXPECT_FLOAT_EQ(y[1], -1.0);
   EXPECT_FLOAT_EQ(y[2], 2.0);
   EXPECT_FLOAT_EQ(y[3], -1.0);
   EXPECT_FLOAT_EQ(y[4], 1.0);
   EXPECT_FLOAT_EQ(y[5], -2.0);
}

TEST(RBDT, BatchMultiColumnInput)
{
   const auto maxDepth = 1;
   const auto numInputs = 2;
   const auto numTrees = 1;
   WriteModel("myModel", "TestRBDT5.root", "identity", {0}, {0}, {0.0, 1.0, -1.0}, {maxDepth}, {numTrees}, {numInputs},
              {1});

   RBDT<> bdt("myModel", "TestRBDT5.root");
   // Two events with two features each, both events are identical: {-999.0, 999.0}.
   const std::vector<float> x = {-999.0, 999.0, -999.0, 999.0};
   auto y = bdt.Compute(x, 2);
   EXPECT_EQ(y.size(), 2u);
   EXPECT_FLOAT_EQ(y[0], 1.0);
   EXPECT_FLOAT_EQ(y[1], 1.0);
}
