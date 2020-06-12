// Tests for the RooAbsCollection and derived classes
// Authors: Stephan Hageboeck, CERN  05/2020
#include "RooArgSet.h"

#include "gtest/gtest.h"

/// ROOT-10845 IsOnHeap() always returned false.
TEST(RooArgSet, IsOnHeap) {
  auto setp = new RooArgSet();
  EXPECT_TRUE(setp->IsOnHeap());

  RooArgSet setStack;
  EXPECT_FALSE(setStack.IsOnHeap());
}
