#include "ROOT/BitUtils.hxx"

#include "gtest/gtest.h"

using namespace ROOT::Internal;

TEST(BitUtils, LeadingZeroes32)
{
   EXPECT_EQ(LeadingZeroes(0), 32);
   EXPECT_EQ(LeadingZeroes(~0), 0);
   EXPECT_EQ(LeadingZeroes(0xF000'0000), 0);
   EXPECT_EQ(LeadingZeroes(0x0000'F040), 16);
   EXPECT_EQ(LeadingZeroes(0x0000'0003), 30);
   EXPECT_EQ(LeadingZeroes(0x000F'F000), 12);
}

TEST(BitUtils, LeadingZeroes64)
{
   EXPECT_EQ(LeadingZeroes(0ull), 64);
   EXPECT_EQ(LeadingZeroes(~0ull), 0);
   EXPECT_EQ(LeadingZeroes(0xF000'0000'0000'0000ull), 0);
   EXPECT_EQ(LeadingZeroes(0x0000'F000'1000'1000ull), 16);
   EXPECT_EQ(LeadingZeroes(0x0000'0000'0000'0003ull), 62);
   EXPECT_EQ(LeadingZeroes(0x0000'000F'F000'0000ull), 28);
}

TEST(BitUtils, TrailingZeroes32)
{
   EXPECT_EQ(TrailingZeroes(0), 32);
   EXPECT_EQ(TrailingZeroes(~0), 0);
   EXPECT_EQ(TrailingZeroes(0xF000'0000), 28);
   EXPECT_EQ(TrailingZeroes(0x0000'F040), 6);
   EXPECT_EQ(TrailingZeroes(0x0000'0003), 0);
   EXPECT_EQ(TrailingZeroes(0x000F'F000), 12);
}

TEST(BitUtils, TrailingZeroes64)
{
   EXPECT_EQ(TrailingZeroes(0ull), 64);
   EXPECT_EQ(TrailingZeroes(~0ull), 0);
   EXPECT_EQ(TrailingZeroes(0xF000'0000'0000'0000ull), 60);
   EXPECT_EQ(TrailingZeroes(0x0000'F000'1000'1000ull), 12);
   EXPECT_EQ(TrailingZeroes(0x0000'0000'0000'0003ull), 0);
   EXPECT_EQ(TrailingZeroes(0x0000'000F'F000'0000ull), 28);
}
