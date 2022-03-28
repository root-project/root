#include "gtest/gtest.h"

#include "Fit/DataRange.h"

//Overlap Range
TEST(Range, Overlap)
{
  ROOT::Fit::DataRange range;
  range.AddRange(0,5);                                 
  range.AddRange(2,3);
  
  EXPECT_EQ(range.Size(), 1);

  EXPECT_EQ(range(0,0).first, 0);
  EXPECT_EQ(range(0,0).second, 5);

  range.AddRange(-1,6);
  EXPECT_EQ(range.Size(), 1);

  EXPECT_EQ(range(0,0).first, -1);
  EXPECT_EQ(range(0,0).second, 6);

  range.AddRange(-2,4);
  EXPECT_EQ(range.Size(), 1);

  EXPECT_EQ(range(0,0).first, -2);
  EXPECT_EQ(range(0,0).second, 6);

  range.AddRange(5,7);
  EXPECT_EQ(range.Size(), 1);

  EXPECT_EQ(range(0,0).first, -2);
  EXPECT_EQ(range(0,0).second, 7);

  range.AddRange(20,25);
  EXPECT_EQ(range.Size(), 2);

  EXPECT_EQ(range(0,0).first, -2);
  EXPECT_EQ(range(0,0).second, 7);

  EXPECT_EQ(range(0,1).first, 20);
  EXPECT_EQ(range(0,1).second, 25);
  
  range.AddRange(24,26);
  EXPECT_EQ(range.Size(), 2);
  EXPECT_EQ(range(0,1).first, 20);
  EXPECT_EQ(range(0,1).second, 26);
  
  range.AddRange(19,20);
  EXPECT_EQ(range(0,1).first, 19);
  EXPECT_EQ(range(0,1).second, 26);

  range.AddRange(6,20);
  EXPECT_EQ(range.Size(), 1);
  EXPECT_EQ(range(0,0).first, -2);
  EXPECT_EQ(range(0,0).second, 26);
}
