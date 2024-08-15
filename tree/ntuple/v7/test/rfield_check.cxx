#include <ROOT/RField.hxx>

#include "CustomStruct.hxx"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

class NoDict {};

TEST(RField, Check)
{
   using ROOT::Experimental::RFieldBase;

   auto report = RFieldBase::Check("f", "CustomStruct");
   EXPECT_TRUE(report.empty());

   report = RFieldBase::Check("f", "");
   EXPECT_EQ(1u, report.size());
   EXPECT_EQ("f", report[0].fFieldName);
   EXPECT_EQ("", report[0].fTypeName);
   EXPECT_THAT(report[0].fErrMsg, testing::HasSubstr("no type name"));

   report = RFieldBase::Check("f", "std::array<>");
   EXPECT_EQ(1u, report.size());
   EXPECT_EQ("f", report[0].fFieldName);
   EXPECT_EQ("std::array<>", report[0].fTypeName);
   EXPECT_THAT(report[0].fErrMsg, testing::HasSubstr("exactly two elements"));

   report = RFieldBase::Check("f", "NoDict");
   EXPECT_EQ(1u, report.size());
   EXPECT_EQ("f", report[0].fFieldName);
   EXPECT_EQ("NoDict", report[0].fTypeName);
   EXPECT_THAT(report[0].fErrMsg, testing::HasSubstr("unknown type"));

   report = RFieldBase::Check("f", "Unsupported");
   EXPECT_EQ(2u, report.size());
   EXPECT_EQ("f.timestamp", report[0].fFieldName);
   EXPECT_THAT(report[0].fTypeName, testing::HasSubstr("chrono::time_point"));
   EXPECT_THAT(report[0].fErrMsg, testing::HasSubstr("unknown type"));
   EXPECT_EQ("f.rd", report[1].fFieldName);
   EXPECT_THAT(report[1].fTypeName, testing::HasSubstr("random_device"));
   EXPECT_THAT(report[1].fErrMsg, testing::HasSubstr("unknown type"));

   report = RFieldBase::Check("f", "long double");
   EXPECT_EQ(1u, report.size());
   EXPECT_EQ("f", report[0].fFieldName);
   EXPECT_EQ("long double", report[0].fTypeName);
   EXPECT_THAT(report[0].fErrMsg, testing::HasSubstr("unknown type"));
}
