// Tests for the RooAbsCollection and derived classes
// Authors: Stephan Hageboeck, CERN  05/2020
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooHelpers.h"

#include "gtest/gtest.h"

/// ROOT-10845 IsOnHeap() always returned false.
TEST(RooArgSet, IsOnHeap) {
  auto setp = new RooArgSet();
  EXPECT_TRUE(setp->IsOnHeap());

  RooArgSet setStack;
  EXPECT_FALSE(setStack.IsOnHeap());
}

TEST(RooArgSet, VariadicTemplateConstructor) {
  RooRealVar x("x", "x", 0.);
  RooRealVar y("y", "y", 0.);
  RooArgSet theSet(x, "Hallo");
  EXPECT_EQ(theSet.size(), 1);
  EXPECT_STREQ(theSet.GetName(), "Hallo");

  RooArgSet theSet2(x, y, "Hallo2");
  EXPECT_EQ(theSet2.size(), 2);
  EXPECT_STREQ(theSet2.GetName(), "Hallo2");
}

TEST(RooArgSet, SubscriptOperator) {
  RooRealVar x("x", "x", 0.);
  RooRealVar y("y", "y", 0.);
  RooArgSet theSet(x, y);

  EXPECT_EQ(theSet[0], &x);
  EXPECT_EQ(&theSet["x"], &x);

  // Silence error to come next:
  RooHelpers::HijackMessageStream hijack(RooFit::ERROR, RooFit::InputArguments);
  EXPECT_THROW(theSet[nullptr], std::invalid_argument);
}
