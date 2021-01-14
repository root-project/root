#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "TString.h"

TEST(TString)
{
   TString p("Test", -5);
   EXPECT_STREQ("", p);
   TString p("Test", 1);
   EXPECT_STREQ("T", p);
}
