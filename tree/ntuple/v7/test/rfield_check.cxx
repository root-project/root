#include <ROOT/RField.hxx>

#include <tuple>

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
   auto [fieldName, typeName, errMsg] = report[0];
   EXPECT_EQ("f", fieldName);
   EXPECT_EQ("", typeName);
   EXPECT_THAT(errMsg, testing::HasSubstr("no type name"));

   report = RFieldBase::Check("f", "std::array<>");
   EXPECT_EQ(1u, report.size());
   std::tie(fieldName, typeName, errMsg) = report[0];
   EXPECT_EQ("f", fieldName);
   EXPECT_EQ("std::array<>", typeName);
   EXPECT_THAT(errMsg, testing::HasSubstr("exactly two elements"));

   report = RFieldBase::Check("f", "NoDict");
   EXPECT_EQ(1u, report.size());
   std::tie(fieldName, typeName, errMsg) = report[0];
   EXPECT_EQ("f", fieldName);
   EXPECT_EQ("NoDict", typeName);
   EXPECT_THAT(errMsg, testing::HasSubstr("unknown type"));

   report = RFieldBase::Check("f", "Unsupported");
   EXPECT_EQ(2u, report.size());
   std::tie(fieldName, typeName, errMsg) = report[0];
   EXPECT_EQ("f.timestamp", fieldName);
   EXPECT_THAT(typeName, testing::HasSubstr("chrono::time_point"));
   EXPECT_THAT(errMsg, testing::HasSubstr("unknown type"));
   std::tie(fieldName, typeName, errMsg) = report[1];
   EXPECT_EQ("f.rd", fieldName);
   EXPECT_THAT(typeName, testing::HasSubstr("random_device"));
   EXPECT_THAT(errMsg, testing::HasSubstr("unknown type"));
}
