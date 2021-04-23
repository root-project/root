#include "ntuple_test.hxx"

TEST(RNTuple, TypeName) {
   EXPECT_STREQ("float", ROOT::Experimental::RField<float>::TypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>",
                ROOT::Experimental::RField<std::vector<std::string>>::TypeName().c_str());
   EXPECT_STREQ("CustomStruct",
                ROOT::Experimental::RField<CustomStruct>::TypeName().c_str());
}


TEST(RNTuple, CreateField)
{
   auto field = RFieldBase::Create("test", "vector<unsigned int>").Unwrap();
   EXPECT_STREQ("std::vector<std::uint32_t>", field->GetType().c_str());
   auto value = field->GenerateValue();
   field->DestroyValue(value);
}

TEST(RNTuple, Int64_t)
{
   auto field = RField<std::int64_t>("int64");
   auto otherField = RFieldBase::Create("test", "std::int64_t").Unwrap();
}

TEST(RNTuple, Int8_t)
{
   auto field = RField<std::int8_t>("int8");
   auto otherField = RFieldBase::Create("test", "std::int8_t").Unwrap();
   // does not compile
   // -- RField.hxx:450:39: error: static assertion failed: no I/O support for this basic C++ type
   // auto charField = RField<char>("char");
}

TEST(RNTuple, Int16_t)
{
   auto field = RField<std::int16_t>("int16");
   auto otherField = RFieldBase::Create("test", "std::int16_t").Unwrap();
   ASSERT_EQ("std::int16_t", RFieldBase::Create("myShort", "Short_t").Unwrap()->GetType());
}

TEST(RNTuple, UInt16_t)
{
   auto field = RField<std::uint16_t>("uint16");
   auto otherField = RFieldBase::Create("test", "std::uint16_t").Unwrap();
   ASSERT_EQ("std::uint16_t", RFieldBase::Create("myUShort", "UShort_t").Unwrap()->GetType());
}

TEST(RNTuple, UnsupportedStdTypes)
{
   try {
      auto field = RFieldBase::Create("pair_field", "std::pair<int, float>").Unwrap();
      FAIL() << "should not be able to make a std::pair field";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("std::pair<int, float> is not supported"));
   }
   try {
      auto field = RField<std::pair<int, float>>("pair_field");
      FAIL() << "should not be able to make a std::pair field";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("pair<int,float> is not supported"));
   }
   try {
      auto field = RField<std::weak_ptr<int>>("weak_ptr");
      FAIL() << "should not be able to make a std::weak_ptr field";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("weak_ptr<int> is not supported"));
   }
   try {
      auto field = RField<std::vector<std::weak_ptr<int>>>("weak_ptr_vec");
      FAIL() << "should not be able to make a std::vector<std::weak_ptr> field";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("weak_ptr<int> is not supported"));
   }
}
