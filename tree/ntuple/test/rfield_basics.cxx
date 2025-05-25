#include "ntuple_test.hxx"

class NoDict {};

TEST(RField, Check)
{
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
   EXPECT_THAT(report[0].fErrMsg, testing::HasSubstr("is not supported"));
   EXPECT_EQ("f.rd", report[1].fFieldName);
   EXPECT_THAT(report[1].fTypeName, testing::HasSubstr("random_device"));
   EXPECT_THAT(report[1].fErrMsg, testing::HasSubstr("is not supported"));

   report = RFieldBase::Check("f", "long double");
   EXPECT_EQ(1u, report.size());
   EXPECT_EQ("f", report[0].fFieldName);
   EXPECT_EQ("long double", report[0].fTypeName);
   EXPECT_THAT(report[0].fErrMsg, testing::HasSubstr("unknown type"));
}

TEST(RField, ValidNaming)
{
   try {
      RFieldBase::Create("x.y", "float").Unwrap();
      FAIL() << "creating a field with an invalid name should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("name 'x.y' cannot contain character '.'"));
   }

   auto field = RFieldBase::Create("x", "float").Unwrap();

   try {
      field->Clone("x.y");
      FAIL() << "cloning a field with an invalid name should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("name 'x.y' cannot contain character '.'"));
   }
}
