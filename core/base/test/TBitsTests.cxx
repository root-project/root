#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "TBits.h"

TEST(TBits, CountBits)
{
   TBits b(50);
   b.SetBitNumber(1);
   EXPECT_EQ( b.CountBits(), 1 );
   EXPECT_EQ( b.CountBits(2), 0 );
   b.SetBitNumber(7);
   EXPECT_EQ( b.CountBits(), 2);
   EXPECT_EQ( b.CountBits(2), 1 );
   EXPECT_EQ( b.CountBits(8), 0);
}
