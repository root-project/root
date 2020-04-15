// Tests for the RooDataSet
// Authors: Stephan Hageboeck, CERN  04/2020

#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooHelpers.h"
#include "TTree.h"

#include "gtest/gtest.h"

/// ROOT-10676
/// The RooDataSet warns that it's not using all variables if the selection string doesn't
/// make use of all variables. Although true, the user has no way to suppress this.
TEST(RooDataSet, ImportFromTreeWithCut)
{
  RooHelpers::HijackMessageStream hijack(RooFit::INFO, RooFit::InputArguments);

  TTree tree("tree", "tree");
  double thex, they;
  tree.Branch("x", &thex);
  tree.Branch("y", &they);
  tree.Branch("z", &they);
  thex = -0.337;
  they = 1.;
  tree.Fill();

  thex = 0.337;
  they = 1.;
  tree.Fill();

  thex = 1.337;
  they = 1.;
  tree.Fill();

  RooRealVar x("x", "x", 0);
  RooRealVar y("y", "y", 0);
  RooRealVar z("z", "z", 0);
  RooDataSet data("data", "data", &tree, RooArgSet(x, y, z), "x>y");

  EXPECT_TRUE(hijack.str().empty()) << "Messages issued were: " << hijack.str();
  EXPECT_EQ(data.numEntries(), 1);

  RooRealVar* theX = dynamic_cast<RooRealVar*>(data.get(0)->find("x"));
  ASSERT_NE(theX, nullptr);
  EXPECT_FLOAT_EQ(theX->getVal(), 1.337);
}
