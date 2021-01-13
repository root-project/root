#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "TString.h"

TEST(TString)
{
	TString n("testNeg", -1);
	TString p("testPos", 1);
	EXPECT_STREQ("", n);
	EXPECT_STREQ("te", p);
}
