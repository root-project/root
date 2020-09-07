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

  std::cout << "Call [0]." << std::endl;
  EXPECT_EQ(theSet[0], &x);
  std::cout << "Call [1]." << std::endl;
  EXPECT_EQ(theSet[1], &y);

  TString::doPrint() = true;
  TString testString("thisIsTheTestString");
  std::cout << "Call [x]." << std::endl;
  EXPECT_EQ(&theSet["x"], &x);
  std::cout << "Call [x] finished." << std::endl;

  // Silence error to come next:
  RooHelpers::HijackMessageStream hijack(RooFit::ERROR, RooFit::InputArguments);
  std::cout << "Call [nullptr]." << std::endl;
  EXPECT_THROW(theSet[nullptr], std::invalid_argument);
}

TEST(RooArgSet, FromVector) {
  std::vector<RooRealVar> vars;
  vars.emplace_back("x", "x", 0.);
  vars.emplace_back("y", "y", 0.);

  RooArgSet theSet(vars.begin(), vars.end(), "Hallo");
  EXPECT_EQ(theSet.size(), 2);
  EXPECT_STREQ(theSet.GetName(), "Hallo");
}
