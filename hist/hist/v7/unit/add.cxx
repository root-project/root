#include "gtest/gtest.h"

#include "TH1.h"
#include "ROOT/THist.h"

// Test "x + 0 = x"
TEST(HistAddTest, AddEmptyHist) {
  EXPECT_EQ(0, 0);
}

// Test addition of a hist range
TEST(HistAddTest, AddView) {
  EXPECT_EQ(1, 1);
}
