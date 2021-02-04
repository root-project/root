#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "TString.h"

TEST(TString, Basics)
{
   TString n("Test", -5);
   EXPECT_STREQ("", n);
   TString p("Test", 1);
   EXPECT_STREQ("T", p);
   TString a = "test";
   a.Append("s", -5);
   EXPECT_STREQ("test", a);
}
