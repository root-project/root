#include "gtest/gtest.h"
#include "TDatime.h"

TEST(TDatime, Constructors)
{
   TDatime d1;
   EXPECT_GT(d1.GetYear(), 1994);

   TDatime d2(2023, 10, 5, 12, 30, 45);
   EXPECT_EQ(2023, d2.GetYear());
   EXPECT_EQ(10, d2.GetMonth());
   EXPECT_EQ(5, d2.GetDay());
   EXPECT_EQ(12, d2.GetHour());
   EXPECT_EQ(30, d2.GetMinute());
   EXPECT_EQ(45, d2.GetSecond());

   TDatime d3("2023-10-05 12:30:45");
   EXPECT_EQ(d2.Get(), d3.Get());
}

TEST(TDatime, Operators)
{
   TDatime d1(2023, 10, 5, 12, 30, 45);
   TDatime d2(2023, 10, 5, 12, 30, 45);
   TDatime d3(2023, 10, 6, 12, 30, 45);

   EXPECT_TRUE(d1 == d2);
   EXPECT_FALSE(d1 != d2);
   EXPECT_TRUE(d1 < d3);
   EXPECT_TRUE(d3 > d1);
   EXPECT_TRUE(d1 <= d2);
   EXPECT_TRUE(d1 >= d2);
}

TEST(TDatime, Getters)
{
   TDatime d1(2023, 10, 5, 12, 30, 45);
   EXPECT_EQ(20231005, d1.GetDate());
   EXPECT_EQ(123045, d1.GetTime());
}

TEST(TDatime, DateArithmetic)
{
   Int_t globalDay = TDatime::GetGlobalDayFromDate(20231005);
   Int_t date = TDatime::GetDateFromGlobalDay(globalDay);
   EXPECT_EQ(20231005, date);
}

