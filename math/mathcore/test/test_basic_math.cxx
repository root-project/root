#include "gtest/gtest.h"
#include "TMath.h"   // ROOT math utilities

// Basic tests for a couple of TMath functions
TEST(MathCoreTest, SqrtFunction) {
    double val = 9.0;
    double out = TMath::Sqrt(val);
    EXPECT_NEAR(out, 3.0, 1e-12);
}

TEST(MathCoreTest, LogFunction) {
    double val = 2.71828182845904523536; // approx e
    double out = TMath::Log(val);
    EXPECT_NEAR(out, 1.0, 1e-12);
}