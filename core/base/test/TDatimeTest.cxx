#include "gtest/gtest.h"
#include "TDatime.h"

TEST(TDatimeTest, DefaultConstructor) {
    TDatime dt(2025, 1, 1, 12, 0, 0);
    EXPECT_EQ(dt.GetYear(), 2025);
    EXPECT_EQ(dt.GetMonth(), 1);
    EXPECT_EQ(dt.GetDay(), 1);
    EXPECT_EQ(dt.GetHour(), 12);
    EXPECT_EQ(dt.GetMinute(), 0);
    EXPECT_EQ(dt.GetSecond(), 0);
}

TEST(TDatimeTest, SetTime) {
    TDatime dt;
    dt.Set(2022, 11, 23, 10, 15, 30);
    EXPECT_EQ(dt.GetYear(), 2022);
    EXPECT_EQ(dt.GetMonth(), 11);
    EXPECT_EQ(dt.GetDay(), 23);
    EXPECT_EQ(dt.GetHour(), 10);
    EXPECT_EQ(dt.GetMinute(), 15);
    EXPECT_EQ(dt.GetSecond(), 30);
}
